import hashlib
import json

import pytest

from model_pipeline.core.errors import ExportError
from model_pipeline.steps import export
from model_pipeline.steps.export import (
    ExportModel, build_manifest, convert, folder_sha256, harnet10_fingerprint, sha256_of, write_bundle)
from model_pipeline.tracking.memory_tracker import MemoryTracker
from tests.helpers import make_config

HARNET_SHA = hashlib.sha256(b"fake harnet10 weights").hexdigest()
EXPORT_CFG = {"family": "parkinsons-risk", "device": {"enabled": False, "required": False, "format": None}}
STATS = {"rows": 10, "features": {}}


def saved_model(folder, harnet_sha=None):
    """A folder laid out as Model.save writes it: manifest.json and weights/."""
    (folder / "weights").mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({
        "family": "window_logistic", "labels": ["state_now", "sleep_now"], "window_spec": {"spec_version": 1}}),
        encoding="utf-8")
    (folder / "weights" / "model.pkl").write_bytes(b"fitted weights")
    if harnet_sha:
        (folder / "weights" / "training.json").write_text(json.dumps({"harnet10": harnet_sha}), encoding="utf-8")
    return folder


def candidate(model_dir):
    return {"name": "window_logistic", "family": "window_logistic", "run_id": "train-run-7", "split_hash": "abc",
            "model_dir": str(model_dir), "metrics": {"cv/state_now/off_auc": 0.74}, "gates_passed": True}


def make_step(tmp_path, device=None):
    cfg = make_config(tmp_path)
    if device:
        cfg["export"]["device"].update(device)
    tracker = MemoryTracker()
    return ExportModel(cfg, tracker), tracker


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert sha256_of(path) == hashlib.sha256(b"abc").hexdigest()


def test_folder_sha256_changes_with_any_file(tmp_path):
    model = saved_model(tmp_path / "m")
    before = folder_sha256(model)
    assert folder_sha256(model) == before
    (model / "weights" / "model.pkl").write_bytes(b"other weights")
    assert folder_sha256(model) != before


def test_harnet10_fingerprint_is_read_from_the_saved_model(tmp_path):
    assert harnet10_fingerprint(saved_model(tmp_path / "a", HARNET_SHA)) == HARNET_SHA
    assert harnet10_fingerprint(saved_model(tmp_path / "b")) is None


def test_convert_is_a_placeholder():
    with pytest.raises(ExportError, match="on-device export isn't built"):
        convert("model", "onnx")


def test_manifest_has_what_the_backend_needs(tmp_path):
    model = saved_model(tmp_path / "m", HARNET_SHA)
    manifest = build_manifest(EXPORT_CFG, candidate(model), model)
    assert manifest["family"] == "parkinsons-risk" and manifest["runs_on"] == "server"
    assert manifest["candidate"] == "window_logistic" and manifest["model_family"] == "window_logistic"
    assert manifest["labels"] == ["state_now", "sleep_now"] and manifest["window_spec_version"] == 1
    assert manifest["mlflow_run_id"] == "train-run-7" and manifest["split_hash"] == "abc"
    assert manifest["model_sha256"] == folder_sha256(model)
    assert manifest["harnet10_sha256"] == HARNET_SHA and manifest["device"] is None


def test_write_bundle_replaces_the_last_one(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "old.bin").write_bytes(b"old")
    write_bundle(bundle, saved_model(tmp_path / "m"), {"r": 1}, {"m": 0.7})
    assert sorted(p.name for p in bundle.iterdir()) == ["manifest.json", "metrics.json", "model"]
    assert (bundle / "model" / "manifest.json").is_file()


def test_write_bundle_adds_stats_and_bias_when_given(tmp_path):
    bias = tmp_path / "bias.json"
    bias.write_text('{"gap": 0.05}', encoding="utf-8")
    bundle = write_bundle(tmp_path / "bundle", saved_model(tmp_path / "m"), {}, {}, STATS, bias)
    assert read(bundle / "stats.json") == STATS and read(bundle / "bias.json") == {"gap": 0.05}


def test_run_bundles_the_model_folder_for_the_server(tmp_path):
    step, tracker = make_step(tmp_path)
    bundle = step.run(candidate(saved_model(tmp_path / "m", HARNET_SHA)), STATS)
    assert bundle == tmp_path / "artifacts" / "bundle"
    assert (bundle / "model" / "weights" / "model.pkl").is_file()
    assert not (bundle / "harnet10").exists() and not (bundle / "device").exists()   # licence: never bundled
    assert read(bundle / "manifest.json")["harnet10_sha256"] == HARNET_SHA
    assert read(bundle / "metrics.json") == {"cv/state_now/off_auc": 0.74}
    assert read(bundle / "stats.json") == STATS
    run = tracker.find("export")
    assert run.tags["owner"] == "P4" and run.tags["device_export"] == "false"


def test_run_without_a_saved_model_stops_the_run(tmp_path):
    step, _ = make_step(tmp_path)
    with pytest.raises(ExportError, match="no saved model"):
        step.run(candidate(tmp_path / "missing"))


def test_device_export_on_fails_until_a_converter_exists(tmp_path):
    step, tracker = make_step(tmp_path, device={"enabled": True, "format": "onnx"})
    with pytest.raises(ExportError, match="on-device export isn't built"):
        step.run(candidate(saved_model(tmp_path / "m")))
    assert tracker.find("export").status.value == "FAILED"


def test_device_export_adds_the_phone_file(tmp_path, monkeypatch):
    phone_file = tmp_path / "model.onnx"
    phone_file.write_bytes(b"phone model")
    monkeypatch.setattr(export, "convert", lambda model_dir, fmt: phone_file)
    step, tracker = make_step(tmp_path, device={"enabled": True, "format": "onnx"})
    bundle = step.run(candidate(saved_model(tmp_path / "m")))
    assert (bundle / "device" / "model.onnx").read_bytes() == b"phone model"
    assert read(bundle / "manifest.json")["device"] == {
        "format": "onnx", "file": "model.onnx", "sha256": sha256_of(phone_file)}
    assert tracker.find("export").tags["device_export"] == "true"
