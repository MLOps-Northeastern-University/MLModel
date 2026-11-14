"""Pretrained bodies, kept so a fit with the same pretraining loads it instead of pretraining again.

Pretraining (training.pretrain) is most of a fit's time and depends only on things the diary step's
settings don't touch. So when tuning tries diary settings on the same folds, or train reruns them,
each fold's pretrained body (the movement encoder, token builder and decoder) is made once:

    cache = PretrainedCache.from_environment()            # MODEL_PRETRAIN_CACHE, "off" to disable
    key = cache.key(train_recs, val_recs, feature_keys, material)
    entry = cache.load(key)                                # None on a miss
    cache.save(key, state, history, rng)

The key covers everything that shapes the body, so a stale one can't be reused:
    - the training recordings' contents: their harnet10 features (by the feature cache's key, which
      covers the windows and harnet10's weights), window times, event fields (doses and check-ins),
      patient fields and diary answers (pretraining simulates check-ins from them);
    - the validation recordings the same way (they're only measured, but the history records them);
    - the network and pretraining settings, the dose span and the seed;
    - this package's code (every .py file), PyTorch's version and the device type, since CPU and GPU
      round differently.

An entry also holds the random-number states as pretraining left them, so the diary step after a
hit draws exactly what it would have drawn after pretraining: a cached fit equals an uncached one.

It's a speed-up, never a record: deleting the folder only costs the time to pretrain again. Entries
are written through a temporary file and moved into place, so a crash leaves none half-written.
Each entry is about 1 MB.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import torch

from scripts.utils.logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_FOLDER = Path("artifacts") / "cache" / "pretrained"
FORMAT_VERSION = 1
PACKAGE = Path(__file__).resolve().parent


def code_fingerprint() -> str:
    """A hash of this package's code: any change to it makes every old entry a miss."""
    digest = hashlib.blake2b(digest_size=8)
    for path in sorted(PACKAGE.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def recordings_digest(recordings: Mapping, feature_keys: Mapping[str, Mapping]) -> str:
    """A hash of recordings' contents, participant by participant (features by their cache key)."""
    digest = hashlib.blake2b(digest_size=16)
    for pid in sorted(recordings):
        rec = recordings[pid]
        digest.update(pid.encode("utf-8"))
        digest.update(json.dumps(feature_keys[pid], sort_keys=True).encode("utf-8"))
        for array in (rec.time_s, rec.events, rec.patient):
            digest.update(np.ascontiguousarray(array).tobytes())
        for hour in sorted(rec.answers):
            digest.update(np.int64(hour).tobytes())
            digest.update(np.ascontiguousarray(rec.answers[hour], dtype=np.float32).tobytes())
        digest.update("|".join(rec.labels).encode("utf-8"))
    return digest.hexdigest()


class PretrainedCache:
    """Pretrained bodies in folder, as <key>.pt (weights, history, random states) and <key>.json (what it is)."""

    def __init__(self, folder):
        self.folder = Path(folder)

    @classmethod
    def from_environment(cls) -> Optional["PretrainedCache"]:
        """The cache at MODEL_PRETRAIN_CACHE (default artifacts/cache/pretrained), or None if it's "off"."""
        where = os.environ.get("MODEL_PRETRAIN_CACHE")
        if where is not None and where.strip().lower() in ("off", "0", "false", "none", ""):
            return None
        return cls(where or DEFAULT_FOLDER)

    def key(self, train, val, feature_keys, material: Mapping) -> str:
        """The key of a pretraining: its recordings, settings, code and environment (see the top of this file)."""
        parts = {"format": FORMAT_VERSION, "code": code_fingerprint(), "torch": torch.__version__,
                 "train": recordings_digest(train, feature_keys),
                 "val": recordings_digest(val, feature_keys) if val else None, **material}
        text = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()

    def load(self, key: str) -> Optional[Dict]:
        """The entry {"state", "history", "rng"}, or None if there's none (or it can't be read)."""
        path = self.folder / f"{key}.pt"
        if not path.exists():
            logger.info("no pretrained body cached for this fit (%s): pretraining", key[:12],
                        extra={"event": "pretrain_cache_miss", "key": key})
            return None
        try:
            entry = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:                     # a damaged file is a miss, never a failed fit
            logger.warning("the cached pretrained body %s can't be read (%s): pretraining again", key[:12], e,
                           extra={"event": "pretrain_cache_unreadable", "key": key})
            return None
        if entry.get("format") != FORMAT_VERSION:
            return None
        logger.info("reusing the cached pretrained body %s", key[:12],
                    extra={"event": "pretrain_cache_hit", "key": key})
        return entry

    def save(self, key: str, state: Mapping[str, torch.Tensor], history, rng: Mapping, about: Mapping):
        """Write an entry atomically, or log a warning and return None if it can't be written.
        about goes in <key>.json, for whoever looks in the folder."""
        path = self.folder / f"{key}.pt"
        tmp = path.with_name(f".{key}.{os.getpid()}.tmp")
        info = path.with_suffix(".json")
        tmp_info = info.with_name(f".{key}.{os.getpid()}.json.tmp")
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            torch.save({"format": FORMAT_VERSION, "state": {k: v.detach().cpu() for k, v in state.items()},
                        "history": history, "rng": rng}, tmp)
            os.replace(tmp, path)
            tmp_info.write_text(json.dumps(about, indent=1, default=str), encoding="utf-8")
            os.replace(tmp_info, info)
        except OSError as e:                       # a full disk or a locked file costs the speed-up, not the fit
            for leftover in (tmp, tmp_info):
                leftover.unlink(missing_ok=True)
            logger.warning("couldn't cache the pretrained body %s (%s); the fit goes on", key[:12], e,
                           extra={"event": "pretrain_cache_unwritable", "key": key})
            return None
        logger.info("cached the pretrained body %s", key[:12], extra={"event": "pretrain_cache_saved", "key": key})
        return path


def random_states(rng: np.random.Generator) -> Dict:
    """The numpy generator's and PyTorch's random states, to resume exactly where pretraining left them."""
    states = {"numpy": rng.bit_generator.state, "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_random_states(rng: np.random.Generator, states: Mapping) -> None:
    rng.bit_generator.state = states["numpy"]
    torch.set_rng_state(states["torch"])
    if "cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])


__all__ = ["PretrainedCache", "code_fingerprint", "random_states", "recordings_digest", "restore_random_states"]
