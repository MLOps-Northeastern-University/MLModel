"""Pushes the bundle to the model registry, and rolls back.

Owner: Ayush

The registry keeps every released bundle as <family>/v<N>/ and an active.json
that names the version patients get. Registering copies the bundle in as the
next version and makes it active; rolling back makes the previous version
active again. The active version's metrics.json is what the next run's release
gate (compare_production.py) compares against.

Where it lives: paths.registry_dir, a local folder, on a laptop and in tests;
MODEL_REGISTRY replaces it, and CI sets that to gs://team17-mlops-neu-models, so
releases persist between runs. fsspec (with gcsfs for gs://) reads and writes
both the same way.

    python -m model_pipeline.steps.register status      the versions and the active one
    python -m model_pipeline.steps.register rollback    the version before the active one

PLACEHOLDER: registry.backend_url is empty, because the Backend has no endpoint
for new model versions yet, so telling the Backend is skipped with a warning.
"""


import argparse
import json
import os
import sys
from pathlib import Path

import fsspec

from model_pipeline.core.errors import PipelineError
from model_pipeline.utils.config import load_config
from model_pipeline.utils.logging_setup import get_logger

logger = get_logger(__name__)


def _fs(location):
    """The filesystem and the path inside it, for a local folder or a gs:// path."""
    return fsspec.core.url_to_fs(str(location))


def _join(location, *parts):
    """location/part/part, the same way for a folder and a gs:// path."""
    return "/".join([str(location).rstrip("/"), *parts])


def _read_json(location):
    fs, path = _fs(location)
    with fs.open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(obj, location):
    fs, path = _fs(location)
    fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
    with fs.open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def versions(family_dir):
    """The released version numbers, oldest first: v1, v2 ... -> [1, 2, ...]."""
    fs, root = _fs(family_dir)
    if not fs.isdir(root):
        return []
    names = [p.rstrip("/").rsplit("/", 1)[-1] for p in fs.ls(root, detail=False)]
    return sorted(int(n[1:]) for n in names if n.startswith("v") and n[1:].isdigit())


def next_version(family_dir):
    """The number the next release gets."""
    return max(versions(family_dir), default=0) + 1


def active_version(family_dir):
    """The version patients get, or None before the first release."""
    location = _join(family_dir, "active.json")
    fs, path = _fs(location)
    return _read_json(location)["version"] if fs.exists(path) else None


def set_active(family_dir, version):
    """Make version the one patients get."""
    if version not in versions(family_dir):
        raise PipelineError(f"version {version} isn't in the registry")
    _write_json({"version": version}, _join(family_dir, "active.json"))


def active_metrics(family_dir):
    """The active version's metrics, for the release gate, or None before the first release."""
    version = active_version(family_dir)
    if version is None:
        return None
    return _read_json(_join(family_dir, f"v{version}", "metrics.json"))


def publish(bundle_dir, family_dir, version):
    """Upload the bundle as v<version>, with its version and model_id in its manifest."""
    bundle_dir = Path(bundle_dir)
    target = _join(family_dir, f"v{version}")
    fs, target_path = _fs(target)
    if fs.exists(target_path):
        raise PipelineError(f"{target} already exists; a released version is never overwritten")
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["version"] = version
    manifest["model_id"] = f"{manifest['family']}-v{version}"
    for path in sorted(p for p in bundle_dir.rglob("*") if p.is_file()):
        if path == bundle_dir / "manifest.json":
            continue
        remote = _join(target_path, path.relative_to(bundle_dir).as_posix())
        fs.makedirs(remote.rsplit("/", 1)[0], exist_ok=True)
        fs.put_file(str(path), remote)
    _write_json(manifest, _join(target, "manifest.json"))
    return manifest


def notify_backend(backend_url, manifest):
    """PLACEHOLDER: tell the Backend about the new version, once it has an endpoint."""
    if not backend_url:
        logger.warning("Backend not told about %s: registry.backend_url isn't set", manifest["model_id"])
        return False
    raise NotImplementedError("registering with the Backend waits on its endpoint")


class RegisterModel:
    """Releases a bundle as the next version, and rolls back."""

    def __init__(self, cfg, tracker):
        # MODEL_REGISTRY (set in CI to gs://team17-mlops-neu-models) replacing the local folder.
        root = os.environ.get("MODEL_REGISTRY") or cfg["resolved_paths"]["registry_dir"]
        self.family_dir = _join(root, cfg["export"]["family"])
        self.backend_url = cfg["registry"]["backend_url"]
        self.tracker = tracker

    def run(self, bundle_dir):
        """Release the bundle as the next version and make it active. Returns the version."""
        with self.tracker.run("register", tags={"owner": "P4"}):
            version = next_version(self.family_dir)
            manifest = publish(Path(bundle_dir), self.family_dir, version)
            set_active(self.family_dir, version)
            notify_backend(self.backend_url, manifest)
            self.tracker.set_tags({"model_id": manifest["model_id"], "version": version})
            logger.info("Released %s as the active version", manifest["model_id"])
            return version

    def rollback(self):
        """Make the version before the active one active again. Returns it."""
        active = active_version(self.family_dir)
        earlier = [v for v in versions(self.family_dir) if active is not None and v < active]
        if not earlier:
            raise PipelineError("nothing to roll back to: no version before the active one")
        set_active(self.family_dir, earlier[-1])
        logger.warning("Rolled back from v%d to v%d", active, earlier[-1])
        return earlier[-1]


def main(argv=None):
    """python -m model_pipeline.steps.register status | rollback"""
    parser = argparse.ArgumentParser(description="Show the model registry, or roll back one version")
    parser.add_argument("command", choices=["status", "rollback"])
    parser.add_argument("--config", default=None, help="Path to a config YAML")
    args = parser.parse_args(argv)

    register = RegisterModel(load_config(args.config), tracker=None)
    try:
        if args.command == "rollback":
            register.rollback()
        print(f"{register.family_dir}: versions {versions(register.family_dir)}, "
              f"active v{active_version(register.family_dir)}")
    except PipelineError as exc:
        print(f"Registry: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
