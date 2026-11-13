"""Exports the selected model and builds the release bundle.

Owner: Ayush

Builds artifacts/bundle/ for register.py: the model file, a manifest (what the
phone and Backend need to run and check it), the model's metrics (which the
next run's release gate compares against), and its training statistics for
drift.

The conversion to the phone format waits on the runtime spike (MODEL.md, "On
the phone": ExecuTorch, ONNX Runtime or LiteRT). Until then, convert() raises
ExportError and the rest of the bundle is built from an already exported file.
"""

import hashlib
import shutil
from pathlib import Path

from scripts.core.errors import ExportError
from scripts.utils.file_io import write_json
from scripts.utils.logging_setup import get_logger

logger = get_logger(__name__)


def sha256_of(path):
    """The file's SHA-256, as hex: the phone checks it after downloading."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(model, fmt):
    """PLACEHOLDER: the model in the phone format, once the runtime is chosen."""
    raise ExportError(f"no converter for format {fmt!r} yet: waiting on the runtime spike")


def build_manifest(export_cfg, candidate, model_path, run_id):
    """What the phone and Backend need to run and check the model.

    The version and model_id are added by register.py, which owns the versions.
    """
    return {
        "family": export_cfg["family"],
        "format": export_cfg["format"],
        "feature_spec_version": export_cfg["feature_spec_version"],
        "inputs": candidate.get("inputs", []),
        "output": {"name": export_cfg["output_name"], "nudge_above": export_cfg["nudge_above"]},
        "test_cases": candidate.get("test_cases", []),
        "file": model_path.name,
        "sha256": sha256_of(model_path),
        "size_bytes": model_path.stat().st_size,
        "mlflow_run_id": run_id,
    }


def write_bundle(bundle_dir, model_path, manifest, metrics, training_stats):
    """Write the bundle folder for register.py, replacing any old one."""
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    shutil.copy2(model_path, bundle_dir / model_path.name)
    write_json(manifest, bundle_dir / "manifest.json")
    write_json(metrics, bundle_dir / "metrics.json")
    write_json(training_stats, bundle_dir / "stats.json")
    return bundle_dir


class ExportModel:
    """Builds the release bundle for the selected model."""

    def __init__(self, cfg, tracker):
        self.export_cfg = cfg["export"]
        self.bundle_dir = Path(cfg["resolved_paths"]["artifacts_dir"]) / "bundle"
        self.tracker = tracker

    def run(self, candidate, model_path, training_stats):
        """Write the bundle and return its folder.

        Args:
            candidate: The selected model from select.py. Its metrics go in
                metrics.json, and its run_id, if any, links back to training.
            model_path: The model file, already in the phone format.
            training_stats: The training inputs' statistics, for drift.

        Raises:
            ExportError: The model file doesn't exist.
        """
        model_path = Path(model_path)
        with self.tracker.run("export", tags={"owner": "P4"}):
            if not model_path.is_file():
                raise ExportError(f"model file {model_path} doesn't exist")
            manifest = build_manifest(self.export_cfg, candidate, model_path, candidate.get("run_id"))
            write_bundle(self.bundle_dir, model_path, manifest, candidate["metrics"], training_stats)
            self.tracker.set_tags({"exported_model": candidate["name"], "sha256": manifest["sha256"]})
            self.tracker.log_artifact(self.bundle_dir, folder="bundle")
            logger.info("Bundle for %s written to %s (%d bytes)", candidate["name"], self.bundle_dir,
                        manifest["size_bytes"])
            return self.bundle_dir
