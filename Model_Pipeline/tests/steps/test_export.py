import hashlib
import json

import pytest

from scripts.core.errors import ExportError
from scripts.steps.export import ExportModel, build_manifest, convert, sha256_of, write_bundle
from scripts.tracking.memory_tracker import MemoryTracker
from tests.helpers import make_config

EXPORT_CFG = {"family": "parkinsons-risk", "format": None, "feature_spec_version": 3,
              "output_name": "off_risk", "nudge_above": 0.8}
CANDIDATE = {"name": "harnet-attn-8h", "metrics": {"cv/state_now/off_auc": 0.74}, "run_id": "train-run-7",
             "inputs": ["window"], "test_cases": [{"inputs": [0.1], "expected": [0.2], "tolerance": 0.01}]}
STATS = {"rows": 236, "features": {}}


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "model.bin"
    path.write_bytes(b"fake model weights")
    return path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_sha256_matches_hashlib(model_file):
    assert sha256_of(model_file) == hashlib.sha256(b"fake model weights").hexdigest()


def test_convert_waits_on_the_runtime_spike():
    with pytest.raises(ExportError, match="runtime spike"):
        convert(object(), None)


def test_manifest_has_what_the_backend_needs(model_file):
    manifest = build_manifest(EXPORT_CFG, CANDIDATE, model_file, "train-run-7")
    assert manifest["family"] == "parkinsons-risk"
    assert manifest["output"] == {"name": "off_risk", "nudge_above": 0.8}
    assert manifest["file"] == "model.bin"
    assert manifest["size_bytes"] == len(b"fake model weights")
    assert manifest["sha256"] == sha256_of(model_file)
    assert manifest["mlflow_run_id"] == "train-run-7"
    assert manifest["test_cases"] == CANDIDATE["test_cases"]


def test_manifest_without_inputs_or_test_cases_uses_empty_lists(model_file):
    manifest = build_manifest(EXPORT_CFG, {"name": "x", "metrics": {}}, model_file, None)
    assert manifest["inputs"] == [] and manifest["test_cases"] == []


def test_write_bundle_writes_every_file(tmp_path, model_file):
    bundle = write_bundle(tmp_path / "bundle", model_file, {"a": 1}, {"m": 0.7}, STATS)
    assert sorted(p.name for p in bundle.iterdir()) == ["manifest.json", "metrics.json", "model.bin", "stats.json"]
    assert read(bundle / "metrics.json") == {"m": 0.7}


def test_write_bundle_removes_files_from_the_last_run(tmp_path, model_file):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "old_model.bin").write_bytes(b"old")
    write_bundle(bundle, model_file, {}, {}, STATS)
    assert not (bundle / "old_model.bin").exists()


def test_run_builds_the_bundle_and_records_it(tmp_path, model_file):
    tracker = MemoryTracker()
    step = ExportModel(make_config(tmp_path), tracker)
    bundle = step.run(CANDIDATE, model_file, STATS)
    assert bundle == tmp_path / "artifacts" / "bundle"
    assert read(bundle / "manifest.json")["mlflow_run_id"] == "train-run-7"
    assert read(bundle / "metrics.json") == CANDIDATE["metrics"]
    run = tracker.find("export")
    assert run.tags["owner"] == "P4" and run.tags["exported_model"] == "harnet-attn-8h"
    assert run.artifacts == [(bundle, "bundle")]


def test_run_without_a_model_file_stops_the_run(tmp_path):
    step = ExportModel(make_config(tmp_path), MemoryTracker())
    with pytest.raises(ExportError, match="doesn't exist"):
        step.run(CANDIDATE, tmp_path / "missing.bin", STATS)
