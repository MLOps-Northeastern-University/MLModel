import pytest

from model_pipeline.models.motor_causal_transformer import harnet


@pytest.fixture(autouse=True)
def _pretrained_cache_in_tmp(tmp_path, monkeypatch):
    """Every test caches pretrained bodies in its own folder, never in the repository's artifacts/."""
    monkeypatch.setenv("MODEL_PRETRAIN_CACHE", str(tmp_path / "pretrained_cache"))


@pytest.fixture(autouse=True)
def _fresh_harnet10_per_test(tmp_path, monkeypatch):
    """harnet10 and window hashes are remembered per process; tests swap in their own fake harnet10s.

    harnet10's folder points at an empty one, so no test ever reads this machine's real copy.
    """
    monkeypatch.setenv("MODEL_HARNET_HUB", str(tmp_path / "no_harnet10_here"))
    harnet._LOADED.clear()
    harnet._WINDOWS_KEYS.clear()
    yield
    harnet._LOADED.clear()
    harnet._WINDOWS_KEYS.clear()
