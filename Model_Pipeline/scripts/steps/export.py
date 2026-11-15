"""Exports the selected model and builds the release bundle.

Owner: Ayush

The model runs on the server (MODEL.md, "On the phone", update of 2026-11-13),
so the bundle is what the server needs to score with it exactly as trained
(Model_Pipeline/README.md, Interface 4):

    artifacts/bundle/
    ├── model/          the candidate's saved model folder (core/model.py, Model.save)
    ├── manifest.json   what the Backend needs: family, candidate, labels, window spec
    │                   version, the model folder's SHA-256 and harnet10's
    ├── metrics.json    its metrics, which the next run's release gate compares against
    ├── stats.json      the training data's statistics, for drift (when given)
    ├── bias.json       P3's bias report (when bias ran)
    └── device/         only with export.device.enabled: the model for the phone

harnet10 isn't bundled: its licence limits redistribution. The server loads its
own pinned copy, and a model refuses one whose SHA-256 differs from training,
so the manifest names the SHA-256 the server needs.

Exporting to the phone is optional and off by default, for testing whether
on-device scoring can work. convert() is a placeholder until a format
(ExecuTorch, ONNX Runtime or LiteRT) is tested.
"""

import hashlib
import json
import shutil
from pathlib import Path

from scripts.core.errors import ExportError
from scripts.utils.file_io import read_json, write_json
from scripts.utils.logging_setup import get_logger

logger = get_logger(__name__)


def sha256_of(path):
    """The file's SHA-256, as hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def folder_sha256(folder):
    """One SHA-256 for a folder: every file's relative path and contents, in a fixed order."""
    folder = Path(folder)
    digest = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        digest.update(path.relative_to(folder).as_posix().encode())
        digest.update(sha256_of(path).encode())
    return digest.hexdigest()


def harnet10_fingerprint(model_dir):
    """The harnet10 weights' SHA-256 the saved model was trained with, or None if it doesn't use harnet10."""
    for info in sorted((Path(model_dir) / "weights").glob("*.json")):
        recorded = json.loads(info.read_text(encoding="utf-8"))
        if isinstance(recorded, dict) and recorded.get("harnet10"):
            return recorded["harnet10"]
    return None


def convert(model_dir, fmt):
    """PLACEHOLDER: the model in a phone format, once one is tested."""
    raise ExportError(f"no converter for format {fmt!r} yet: on-device export isn't built")


def build_manifest(export_cfg, candidate, model_dir):
    """What the Backend needs to run and check the model. register.py adds the version and model_id."""
    saved = read_json(Path(model_dir) / "manifest.json")
    spec = saved.get("window_spec") or {}
    return {
        "family": export_cfg["family"],
        "candidate": candidate["name"],
        "model_family": saved.get("family", candidate.get("family")),
        "labels": saved.get("labels", []),
        "window_spec_version": spec.get("spec_version"),
        "mlflow_run_id": candidate.get("run_id"),
        "split_hash": candidate.get("split_hash"),
        "runs_on": "server",
        "model_sha256": folder_sha256(model_dir),
        "harnet10_sha256": harnet10_fingerprint(model_dir),
        "device": None,
    }


def write_bundle(bundle_dir, model_dir, manifest, metrics, training_stats=None, bias_report=None):
    """Write the bundle folder for register.py, replacing any old one."""
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    shutil.copytree(model_dir, bundle_dir / "model")
    write_json(manifest, bundle_dir / "manifest.json")
    write_json(metrics, bundle_dir / "metrics.json")
    if training_stats is not None:
        write_json(training_stats, bundle_dir / "stats.json")
    if bias_report is not None and Path(bias_report).is_file():
        shutil.copy2(bias_report, bundle_dir / "bias.json")
    return bundle_dir


class ExportModel:
    """Builds the release bundle for the selected model."""

    def __init__(self, cfg, tracker):
        self.export_cfg = cfg["export"]
        self.device_cfg = cfg["export"]["device"]
        artifacts = Path(cfg["resolved_paths"]["artifacts_dir"])
        self.bundle_dir = artifacts / "bundle"
        self.bias_report = artifacts / "bias" / "bias.json"
        self.tracker = tracker

    def run(self, candidate, training_stats=None):
        """Write the bundle for the candidate select.py chose, and return its folder.

        Args:
            candidate: The chosen candidate: its model_dir, metrics, run_id and split_hash.
            training_stats: The training data's statistics for drift, written as stats.json.

        Raises:
            ExportError: The saved model is missing, or the phone export is on and can't be made.
        """
        model_dir = Path(candidate.get("model_dir") or "")
        with self.tracker.run("export", tags={"owner": "P4"}):
            if not (model_dir / "manifest.json").is_file():
                raise ExportError(f"no saved model at {model_dir}")
            manifest = build_manifest(self.export_cfg, candidate, model_dir)
            write_bundle(self.bundle_dir, model_dir, manifest, candidate["metrics"], training_stats,
                         self.bias_report)
            if self.device_cfg["enabled"]:
                self._export_device(model_dir, manifest)
            self.tracker.set_tags({"exported_model": candidate["name"], "model_sha256": manifest["model_sha256"],
                                   "device_export": str(self.device_cfg["enabled"]).lower()})
            self.tracker.log_artifact(self.bundle_dir / "manifest.json", folder="bundle")
            logger.info("Bundle for %s written to %s (device: %s)", candidate["name"], self.bundle_dir,
                        self.device_cfg["enabled"])
            return self.bundle_dir

    def _export_device(self, model_dir, manifest):
        """Optional: the phone file in device/. Fails the run, so a broken phone export is never shipped."""
        fmt = self.device_cfg["format"]
        device_file = Path(convert(model_dir, fmt))
        target = self.bundle_dir / "device"
        target.mkdir()
        shutil.copy2(device_file, target / device_file.name)
        manifest["device"] = {"format": fmt, "file": device_file.name, "sha256": sha256_of(device_file)}
        write_json(manifest, self.bundle_dir / "manifest.json")
