"""harnet10: the frozen network whose features the harnet10 movement encoder reads (movement.py).

harnet10 (Yuan et al. 2024, "Self-supervised learning for human activity recognition using
700,000 person-days of wearable data", npj Digital Medicine) is a ResNet pretrained on UK Biobank
wrist accelerometry. It reads one 10-s window (3 axes x 300 samples at 30 Hz, in g) and gives
1,024 features. Its licence is academic, non-commercial (Oxford's Academic Use Licence); commercial
use needs another.

It never trains, so its features are computed once per window and cached:

    harnet = Harnet10.load(hub_dir, device="cuda")     # a local copy of OxWearables/ssl-wearables
    cache = FeatureCache(folder, harnet)
    features = cache.get("COPS-11", windows)          # (worn windows, 1024) float32, in order

Only worn windows get features. The model never reads the others, and its sequences are worn
windows in order, so row i is the i-th worn window.

A cached participant is reused only if its key matches. The key covers the windows' contents
(a hash of time_s, worn and the worn signal) and harnet10's weights (a SHA-256 of the file). So
changed data or other weights are recomputed, never served stale. A participant's features are
written to a temporary file and moved into place, and its key is written last. So a crash leaves
either a complete entry or one that is recomputed.

Nothing is downloaded. torch.hub runs the local copy's hubconf.py to build the network, so the
copy must be one you trust (the official repository, cloned once).
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch import nn

from scripts.core.data import Windows, WindowSpec
from scripts.core.errors import ConfigError, ModelError
from scripts.utils.logging_setup import get_logger

logger = get_logger(__name__)

FEATURES = 1024
INPUT = {"seconds": 10, "rate_hz": 30, "axes": 3}    # what harnet10 was pretrained on
WEIGHTS = Path("model_check_point") / "mtl_best.mdl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Harnet10:
    """harnet10's frozen feature extractor, and a fingerprint of its weights.

    Args:
        extractor: A module mapping (N, 3, 300) windows to (N, 1024) features (or (N, 1024, 1)).
        fingerprint: Names the weights, such as their SHA-256; part of every cache key.
        batch_size: Windows per forward pass (2,048: about 1 GB of GPU memory).
    """

    def __init__(self, extractor: nn.Module, fingerprint: str, batch_size: int = 2048):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ConfigError(f"harnet10 batch_size must be a positive integer, not {batch_size!r}")
        if not fingerprint:
            raise ConfigError("harnet10 needs a fingerprint of its weights, for the cache key")
        self.extractor = extractor.eval()
        for p in self.extractor.parameters():
            p.requires_grad_(False)
        self.fingerprint, self.batch_size = fingerprint, batch_size

    @classmethod
    def load(cls, hub_dir: Union[str, Path], device: str = "cpu", batch_size: int = 2048) -> "Harnet10":
        """harnet10 from a local copy of OxWearables/ssl-wearables, which holds its weights.

        Raises:
            ConfigError: no copy at hub_dir, or no weights in it.
        """
        hub_dir = Path(hub_dir).expanduser()
        weights = hub_dir / WEIGHTS
        if not (hub_dir / "hubconf.py").exists() or not weights.exists():
            raise ConfigError(f"no harnet10 at {hub_dir}: it needs hubconf.py and {WEIGHTS}")
        model = torch.hub.load(str(hub_dir), "harnet10", source="local", pretrained=True, class_num=5,
                               verbose=False, trust_repo=True)
        return cls(model.feature_extractor.to(device), _sha256(weights), batch_size)

    @property
    def device(self) -> torch.device:
        return next(self.extractor.parameters()).device

    @torch.inference_mode()
    def features(self, signal: np.ndarray, rows: Optional[np.ndarray] = None) -> np.ndarray:
        """harnet10's features of signal's windows (or of signal[rows]), (n, 1024) float32.

        Reads a batch at a time, so signal can be memory-mapped and any length.
        """
        rows = np.arange(len(signal)) if rows is None else np.asarray(rows)
        out = np.empty((len(rows), FEATURES), dtype=np.float32)
        pinned = self.device.type == "cuda"
        for start in range(0, len(rows), self.batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(signal[rows[start:start + self.batch_size]],
                                                          dtype=np.float32))
            if pinned:
                batch = batch.pin_memory()
            x = self.extractor(batch.to(self.device, non_blocking=pinned)).flatten(1)
            if x.shape[1] != FEATURES:
                raise ModelError(f"harnet10 gave {x.shape[1]} features per window, not {FEATURES}")
            out[start:start + len(batch)] = x.float().cpu().numpy()
        if not np.isfinite(out).all():
            raise ModelError("harnet10 gave a feature that isn't finite")
        return out


def check_spec(spec: WindowSpec) -> None:
    """Raise ConfigError unless windows are cut as harnet10 was pretrained on: 10 s at 30 Hz, 3 axes."""
    got = {"seconds": spec.seconds, "rate_hz": spec.rate_hz, "axes": spec.axes}
    if got != INPUT:
        raise ConfigError(f"harnet10 reads windows of {INPUT}, but the window spec is {got}")


# Each windows hash already worked out in this process, for windows read from files (memory-mapped):
# {(signal file, its size, its modification time, a hash of time_s and worn): the hash}. Hashing reads
# every worn window's raw signal, about 3.6 KB a window, and every fit and predict asks again for
# the same patients; a file rewritten in place gets a new modification time, so it's hashed again.
_WINDOWS_KEYS: dict = {}


def windows_key(windows: Windows) -> str:
    """A hash of the windows a participant's features come from: time_s, worn, the worn signal.

    Remembered for the rest of the process when the signal is memory-mapped from a file (see
    _WINDOWS_KEYS); in-memory windows are hashed every time.
    """
    identity = _file_identity(windows)
    if identity is not None and identity in _WINDOWS_KEYS:
        return _WINDOWS_KEYS[identity]
    key = _hash_windows(windows)
    if identity is not None:
        _WINDOWS_KEYS[identity] = key
    return key


def _file_identity(windows: Windows):
    filename = getattr(windows.signal, "filename", None)
    if not filename:
        return None
    try:
        stat = os.stat(filename)
    except OSError:
        return None
    small = hashlib.blake2b(digest_size=16)
    small.update(np.ascontiguousarray(windows.time_s, dtype=np.float64).tobytes())
    small.update(np.asarray(windows.worn, dtype=bool).tobytes())
    return (os.path.abspath(filename), stat.st_size, stat.st_mtime_ns, small.hexdigest())


def _hash_windows(windows: Windows) -> str:
    digest = hashlib.blake2b(digest_size=20)
    worn = np.asarray(windows.worn, dtype=bool)
    digest.update(np.ascontiguousarray(windows.time_s, dtype=np.float64).tobytes())
    digest.update(worn.tobytes())
    rows = np.flatnonzero(worn)
    for start in range(0, len(rows), 8192):                 # a block at a time: signal may be memory-mapped
        digest.update(np.ascontiguousarray(windows.signal[rows[start:start + 8192]], dtype=np.float32).tobytes())
    return digest.hexdigest()


class FeatureCache:
    """harnet10's features per participant, computed once and kept in folder.

        folder/<participant>.npy    (worn windows, 1024) float32
        folder/<participant>.json   its key: the windows' hash, harnet10's fingerprint, the shape
    """

    def __init__(self, folder: Union[str, Path], harnet: Harnet10):
        self.folder, self.harnet = Path(folder), harnet
        self.folder.mkdir(parents=True, exist_ok=True)
        self.keys = {}      # {participant: the key of the features last returned}, which names their contents

    def get(self, participant: str, windows: Windows) -> np.ndarray:
        """The participant's features, from the cache if its key matches, else computed and cached.

        Returns:
            (worn windows, 1024) float32, read-only and memory-mapped from the cache file.
        """
        key = {"windows": windows_key(windows), "harnet10": self.harnet.fingerprint,
               "shape": [int(np.count_nonzero(windows.worn)), FEATURES]}
        self.keys[participant] = key
        features_file, key_file = self.folder / f"{participant}.npy", self.folder / f"{participant}.json"
        if key_file.exists() and features_file.exists():
            try:
                same = json.loads(key_file.read_text(encoding="utf-8")) == key
            except (OSError, ValueError):
                same = False
            if same:
                try:
                    cached = np.load(features_file, mmap_mode="r")
                except (OSError, ValueError):            # a damaged file is recomputed, never a failed fit
                    cached = None
                if cached is not None and cached.shape == tuple(key["shape"]) and cached.dtype == np.float32:
                    return cached
        logger.info("computing harnet10 features for %s (%d worn windows)", participant, key["shape"][0],
                    extra={"event": "harnet_features", "participant": participant, "windows": key["shape"][0]})
        features = self.harnet.features(windows.signal, np.flatnonzero(windows.worn))
        key_file.unlink(missing_ok=True)                      # the entry is invalid until its key is back
        tmp = features_file.with_name(f".{participant}.{os.getpid()}.tmp.npy")
        np.save(tmp, features)
        os.replace(tmp, features_file)
        tmp_key = key_file.with_name(f".{participant}.{os.getpid()}.tmp.json")
        tmp_key.write_text(json.dumps(key), encoding="utf-8")
        os.replace(tmp_key, key_file)
        return np.load(features_file, mmap_mode="r")


# Where harnet10 and its cache are, which isn't part of a model (environment variables):
#     MODEL_HARNET_HUB     the local copy of OxWearables/ssl-wearables
#     MODEL_FEATURE_CACHE  where the features are cached
#     MODEL_DEVICE         cuda or cpu (cuda when there is one)
DEFAULT_HUB = Path.home() / ".cache" / "torch" / "hub" / "OxWearables_ssl-wearables_main"
DEFAULT_CACHE = Path("artifacts") / "cache" / "harnet10"


def default_device() -> str:
    """MODEL_DEVICE, else cuda when there is one, else cpu."""
    return os.environ.get("MODEL_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


# harnet10 as loaded in this process: {(hub folder, device, the weights file's size and time): Harnet10}.
# Loading builds the network through torch.hub and hashes the weights file, and every fit and
# predict would otherwise do it again. A changed weights file is a different key, so it's reloaded.
_LOADED: dict = {}


def feature_cache(device: str, fingerprint: Optional[str] = None) -> FeatureCache:
    """harnet10 and its feature cache, from where the environment says. Every model reading harnet10 uses this.

    harnet10 is loaded once per process (see _LOADED) and its fingerprint is checked every time.

    Args:
        device: Where harnet10 runs when a participant's features aren't cached yet.
        fingerprint: The weights a fitted model was trained with, or None before fitting.

    Raises:
        ModelError: The weights found aren't the ones named by fingerprint.
    """
    hub = Path(os.environ.get("MODEL_HARNET_HUB") or DEFAULT_HUB).expanduser()
    weights = hub / WEIGHTS
    if weights.exists():           # only weights on disk can be told apart, so only those are remembered
        loaded_key = (str(hub.resolve()), str(device), weights.stat().st_size, weights.stat().st_mtime_ns)
        if loaded_key not in _LOADED:
            _LOADED[loaded_key] = Harnet10.load(hub, device=device)
        harnet = _LOADED[loaded_key]
    else:
        harnet = Harnet10.load(hub, device=device)
    if fingerprint is not None and harnet.fingerprint != fingerprint:
        raise ModelError("these harnet10 weights aren't the ones this model was trained with")
    return FeatureCache(os.environ.get("MODEL_FEATURE_CACHE") or DEFAULT_CACHE, harnet)


__all__ = ["DEFAULT_CACHE", "DEFAULT_HUB", "FEATURES", "FeatureCache", "Harnet10", "check_spec", "default_device",
           "feature_cache", "windows_key"]
