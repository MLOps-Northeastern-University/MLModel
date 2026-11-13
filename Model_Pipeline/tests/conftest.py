import pytest


@pytest.fixture(autouse=True)
def _pretrained_cache_in_tmp(tmp_path, monkeypatch):
    """Every test caches pretrained bodies in its own folder, never in the repository's artifacts/."""
    monkeypatch.setenv("MODEL_PRETRAIN_CACHE", str(tmp_path / "pretrained_cache"))
