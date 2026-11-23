import json
import os
import time

import numpy as np
import pytest
import torch
from torch import nn

from model_pipeline.core.data import Windows, WindowSpec
from model_pipeline.core.errors import ConfigError, ModelError
from model_pipeline.models.motor_causal_transformer import harnet as harnet_module
from model_pipeline.models.motor_causal_transformer.harnet import FeatureCache, Harnet10, check_spec, windows_key


class Fake(nn.Module):
    """Stands in for harnet10: (N, 3, 300) -> (N, 1024, 1), and counts the windows it saw."""

    def __init__(self, scale=1.0):
        super().__init__()
        self.proj = nn.Linear(3, 1024)
        self.scale, self.seen = scale, 0

    def forward(self, x):
        self.seen += len(x)
        return (self.proj(x.mean(-1)) * self.scale).unsqueeze(-1)


def harnet(fingerprint="weights-a", batch_size=7, scale=1.0):
    torch.manual_seed(0)
    return Harnet10(Fake(scale), fingerprint, batch_size=batch_size)


def windows(n=40, seed=0):
    rng = np.random.default_rng(seed)
    worn = rng.random(n) > 0.25
    return Windows(time_s=np.arange(n) * 10.0, worn=worn, signal=rng.normal(size=(n, 3, 300)).astype(np.float32))


def test_features_are_the_worn_windows_in_order_in_batches():
    h, w = harnet(), windows()
    want = h.extractor(torch.from_numpy(w.signal[w.worn])).flatten(1).detach().numpy()
    got = h.features(w.signal, np.flatnonzero(w.worn))
    assert got.shape == (w.worn.sum(), 1024) and got.dtype == np.float32
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(harnet(batch_size=1000).features(w.signal, np.flatnonzero(w.worn)), got, atol=1e-6)


def test_the_cache_computes_once_then_reads(tmp_path):
    h, w = harnet(), windows()
    cache = FeatureCache(tmp_path, h)
    first = cache.get("P1", w)
    seen = h.extractor.seen
    again = cache.get("P1", w)
    assert h.extractor.seen == seen                                  # read, not computed
    assert isinstance(again, np.memmap) and not again.flags.writeable
    np.testing.assert_array_equal(first, again)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["P1.json", "P1.npy"]   # no temporary files left


def test_changed_windows_or_weights_are_recomputed(tmp_path):
    w = windows()
    FeatureCache(tmp_path, harnet()).get("P1", w)
    changed = Windows(w.time_s, w.worn, w.signal.copy())
    changed.signal[np.flatnonzero(w.worn)[3], 0, 0] += 1.0            # one sample of one worn window
    h = harnet()
    FeatureCache(tmp_path, h).get("P1", changed)
    assert h.extractor.seen == w.worn.sum()
    other = harnet(fingerprint="weights-b", scale=2.0)
    out = FeatureCache(tmp_path, other).get("P1", changed)
    assert other.extractor.seen == w.worn.sum()
    np.testing.assert_allclose(out, other.features(changed.signal, np.flatnonzero(changed.worn)), atol=1e-6)


def test_an_unworn_windows_signal_doesnt_change_the_key():
    w = windows()
    changed = Windows(w.time_s, w.worn, w.signal.copy())
    changed.signal[np.flatnonzero(~w.worn)[0]] = 9.0
    assert windows_key(changed) == windows_key(w)
    worn_changed = Windows(w.time_s, ~w.worn, w.signal)
    assert windows_key(worn_changed) != windows_key(w)


def test_a_half_written_entry_is_recomputed(tmp_path):
    w = windows()
    FeatureCache(tmp_path, harnet()).get("P1", w)
    (tmp_path / "P1.json").unlink()                                   # crashed before the key was written
    h = harnet()
    FeatureCache(tmp_path, h).get("P1", w)
    assert h.extractor.seen == w.worn.sum()
    (tmp_path / "P1.json").write_text("{not json", encoding="utf-8")  # a broken key
    h = harnet()
    FeatureCache(tmp_path, h).get("P1", w)
    assert h.extractor.seen == w.worn.sum()
    key = json.loads((tmp_path / "P1.json").read_text(encoding="utf-8"))
    assert key["harnet10"] == "weights-a" and key["shape"] == [int(w.worn.sum()), 1024]


def test_no_worn_windows_gives_an_empty_table(tmp_path):
    w = windows()
    none = Windows(w.time_s, np.zeros_like(w.worn), w.signal)
    assert FeatureCache(tmp_path, harnet()).get("P2", none).shape == (0, 1024)


def test_refuses_bad_output_settings_and_specs(tmp_path):
    class Wrong(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return torch.zeros(len(x), 512)

    w = windows()
    with pytest.raises(ModelError, match="1024"):
        Harnet10(Wrong(), "x").features(w.signal)
    with pytest.raises(ConfigError, match="batch_size"):
        Harnet10(Fake(), "x", batch_size=0)
    with pytest.raises(ConfigError, match="fingerprint"):
        Harnet10(Fake(), "")
    with pytest.raises(ConfigError, match="no harnet10"):
        Harnet10.load(tmp_path)
    check_spec(WindowSpec(version=1, seconds=10, rate_hz=30, axes=3))
    with pytest.raises(ConfigError, match="harnet10 reads"):
        check_spec(WindowSpec(version=1, seconds=5, rate_hz=30, axes=3))


def test_a_windows_hash_is_remembered_for_files_and_redone_when_the_file_changes(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "signal.npy"
    np.save(path, rng.normal(size=(20, 3, 300)).astype(np.float32))
    time_s, worn = np.arange(20) * 10.0, np.ones(20, bool)
    first = harnet_module.windows_key(Windows(time_s, worn, np.load(path, mmap_mode="r")))
    assert len(harnet_module._WINDOWS_KEYS) == 1
    assert harnet_module.windows_key(Windows(time_s, worn, np.load(path, mmap_mode="r"))) == first
    changed = rng.normal(size=(20, 3, 300)).astype(np.float32)
    np.save(path, changed)                                     # same size, new contents and time
    os.utime(path, ns=(time.time_ns(), time.time_ns() + 10_000_000))
    assert harnet_module.windows_key(Windows(time_s, worn, np.load(path, mmap_mode="r"))) != first
    in_memory = Windows(time_s, worn, changed)                 # plain arrays: hashed every time, never stored
    harnet_module.windows_key(in_memory)
    assert len(harnet_module._WINDOWS_KEYS) == 2


def test_a_damaged_cached_features_file_is_recomputed(tmp_path):
    cache = harnet_module.FeatureCache(tmp_path, harnet_module.Harnet10(Fake(), "fake-weights"))
    windows = Windows(np.arange(5) * 10.0, np.ones(5, bool), np.random.default_rng(0).normal(
        size=(5, 3, 300)).astype(np.float32))
    good = np.array(cache.get("A", windows))
    (tmp_path / "A.npy").write_bytes(b"truncated")
    np.testing.assert_allclose(np.array(cache.get("A", windows)), good)
