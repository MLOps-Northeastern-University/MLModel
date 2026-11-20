import json

import pytest

from model_pipeline.core.errors import PipelineError
from model_pipeline.steps import register as register_module
from model_pipeline.steps.register import (
    RegisterModel, active_metrics, active_version, next_version, notify_backend, publish, set_active, versions)
from model_pipeline.tracking.memory_tracker import MemoryTracker
from model_pipeline.utils.file_io import write_json
from tests.helpers import make_config


def make_bundle(path, off_auc=0.74):
    """A bundle as export.py writes it."""
    path.mkdir(parents=True)
    (path / "model.bin").write_bytes(b"weights")
    write_json({"family": "parkinsons-risk", "candidate": "window_logistic"}, path / "manifest.json")
    write_json({"cv/state_now/off_auc": off_auc}, path / "metrics.json")
    write_json({"rows": 10}, path / "stats.json")
    return path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def family_dir(tmp_path):
    return tmp_path / "registry" / "parkinsons-risk"


@pytest.fixture
def step(tmp_path):
    tracker = MemoryTracker()
    return RegisterModel(make_config(tmp_path), tracker), tracker


def test_an_empty_registry_has_no_versions(family_dir):
    assert versions(family_dir) == []
    assert next_version(family_dir) == 1
    assert active_version(family_dir) is None
    assert active_metrics(family_dir) is None


def test_versions_sort_as_numbers_and_skip_other_files(family_dir):
    for name in ["v2", "v10", "v9", "vX"]:
        (family_dir / name).mkdir(parents=True)
    (family_dir / "active.json").write_text("{}")
    assert versions(family_dir) == [2, 9, 10]
    assert next_version(family_dir) == 11


def test_publish_adds_the_version_and_model_id(tmp_path, family_dir):
    manifest = publish(make_bundle(tmp_path / "bundle"), family_dir, 3)
    assert manifest["version"] == 3 and manifest["model_id"] == "parkinsons-risk-v3"
    assert read(family_dir / "v3" / "manifest.json") == manifest
    assert (family_dir / "v3" / "model.bin").read_bytes() == b"weights"


def test_a_released_version_is_never_overwritten(tmp_path, family_dir):
    bundle = make_bundle(tmp_path / "bundle")
    publish(bundle, family_dir, 1)
    with pytest.raises(PipelineError, match="never overwritten"):
        publish(bundle, family_dir, 1)


def test_set_active_refuses_a_missing_version(family_dir):
    with pytest.raises(PipelineError, match="isn't in the registry"):
        set_active(family_dir, 5)


def test_active_metrics_are_the_active_versions(tmp_path, family_dir):
    publish(make_bundle(tmp_path / "b1", off_auc=0.70), family_dir, 1)
    publish(make_bundle(tmp_path / "b2", off_auc=0.74), family_dir, 2)
    set_active(family_dir, 1)
    assert active_metrics(family_dir) == {"cv/state_now/off_auc": 0.70}


def test_notify_backend_is_skipped_without_a_url():
    assert notify_backend(None, {"model_id": "parkinsons-risk-v1"}) is False


def test_notify_backend_with_a_url_is_not_built_yet():
    with pytest.raises(NotImplementedError):
        notify_backend("https://backend.example/models", {"model_id": "parkinsons-risk-v1"})


def test_run_releases_versions_in_order(tmp_path, step):
    register, tracker = step
    assert register.run(make_bundle(tmp_path / "b1")) == 1
    assert register.run(make_bundle(tmp_path / "b2")) == 2
    family_dir = tmp_path / "artifacts" / "registry" / "parkinsons-risk"
    assert register.family_dir == str(family_dir)
    assert active_version(family_dir) == 2
    assert tracker.runs["run-2"].tags == {"owner": "P4", "model_id": "parkinsons-risk-v2", "version": "2"}


def test_rollback_steps_back_one_version_at_a_time(tmp_path, step):
    register, _ = step
    for n in range(1, 4):
        register.run(make_bundle(tmp_path / f"b{n}", off_auc=0.70 + n / 100))
    assert register.rollback() == 2
    assert register.rollback() == 1
    assert active_metrics(register.family_dir) == {"cv/state_now/off_auc": 0.71}


def test_rollback_needs_an_earlier_version(tmp_path, step):
    register, _ = step
    with pytest.raises(PipelineError, match="nothing to roll back to"):
        register.rollback()
    register.run(make_bundle(tmp_path / "b1"))
    with pytest.raises(PipelineError, match="nothing to roll back to"):
        register.rollback()


# ---- the registry in a bucket ------------------------------------------------------------
# fsspec's memory:// filesystem stands in for gs://: the same calls, no cloud needed.

@pytest.fixture
def bucket():
    import fsspec
    fs = fsspec.filesystem("memory")
    yield "memory://registry/parkinsons-risk"
    fs.rm("/registry", recursive=True) if fs.exists("/registry") else None


def test_a_bucket_registry_releases_and_rolls_back(tmp_path, bucket):
    publish(make_bundle(tmp_path / "b1", off_auc=0.70), bucket, 1)
    publish(make_bundle(tmp_path / "b2", off_auc=0.74), bucket, 2)
    set_active(bucket, 2)
    assert versions(bucket) == [1, 2] and active_version(bucket) == 2
    assert active_metrics(bucket) == {"cv/state_now/off_auc": 0.74}
    set_active(bucket, 1)
    assert active_metrics(bucket) == {"cv/state_now/off_auc": 0.70}


def test_a_bucket_release_holds_every_file(tmp_path, bucket):
    import fsspec
    publish(make_bundle(tmp_path / "b1"), bucket, 1)
    fs = fsspec.filesystem("memory")
    names = sorted(p.rsplit("/", 1)[-1] for p in fs.find("/registry/parkinsons-risk/v1"))
    assert names == ["manifest.json", "metrics.json", "model.bin", "stats.json"]
    with pytest.raises(PipelineError, match="never overwritten"):
        publish(make_bundle(tmp_path / "b2"), bucket, 1)


def test_model_registry_replaces_the_local_folder(tmp_path, bucket, monkeypatch):
    monkeypatch.setenv("MODEL_REGISTRY", "memory://registry")
    register = RegisterModel(make_config(tmp_path), MemoryTracker())
    assert register.family_dir == bucket
    assert register.run(make_bundle(tmp_path / "b1")) == 1
    assert active_version(bucket) == 1


# ---- the command -------------------------------------------------------------------------

def test_status_prints_the_versions_and_the_active_one(tmp_path, bucket, monkeypatch, capsys):
    monkeypatch.setenv("MODEL_REGISTRY", "memory://registry")
    monkeypatch.setattr(register_module, "load_config", lambda path=None: make_config(tmp_path))
    for n in (1, 2):
        publish(make_bundle(tmp_path / f"b{n}"), bucket, n)
    set_active(bucket, 2)
    assert register_module.main(["status"]) == 0
    assert "versions [1, 2], active v2" in capsys.readouterr().out


def test_rollback_command_steps_back_one_version(tmp_path, bucket, monkeypatch, capsys):
    monkeypatch.setenv("MODEL_REGISTRY", "memory://registry")
    monkeypatch.setattr(register_module, "load_config", lambda path=None: make_config(tmp_path))
    for n in (1, 2):
        publish(make_bundle(tmp_path / f"b{n}"), bucket, n)
    set_active(bucket, 2)
    assert register_module.main(["rollback"]) == 0
    assert active_version(bucket) == 1 and "active v1" in capsys.readouterr().out


def test_rollback_command_fails_with_nothing_to_go_back_to(tmp_path, bucket, monkeypatch, capsys):
    monkeypatch.setenv("MODEL_REGISTRY", "memory://registry")
    monkeypatch.setattr(register_module, "load_config", lambda path=None: make_config(tmp_path))
    assert register_module.main(["rollback"]) == 1
    assert "nothing to roll back to" in capsys.readouterr().err
