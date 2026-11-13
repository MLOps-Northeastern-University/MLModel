"""Pushes the bundle to the model registry, and rolls back.

Owner: Ayush

The registry keeps every released bundle as <family>/v<N>/ and an active.json
that names the version patients get. Registering copies the bundle in as the
next version and makes it active; rolling back makes the previous version
active again. The active version's metrics.json is what the next run's release
gate (compare_production.py) compares against.

PLACEHOLDERS, until the team decides:
- paths.registry_dir is a local folder. It moves to the models bucket or the
  Artifact Registry bundles repository.
- registry.backend_url is empty, because the Backend has no endpoint for new
  model versions yet, so telling the Backend is skipped with a warning.
"""


import shutil
from pathlib import Path

from scripts.core.errors import PipelineError
from scripts.utils.file_io import read_json, write_json
from scripts.utils.logging_setup import get_logger

logger = get_logger(__name__)


def versions(family_dir):
    """The released version numbers, oldest first: v1, v2 ... -> [1, 2, ...]."""
    if not family_dir.is_dir():
        return []
    return sorted(int(p.name[1:]) for p in family_dir.glob("v*") if p.name[1:].isdigit())


def next_version(family_dir):
    """The number the next release gets."""
    return max(versions(family_dir), default=0) + 1


def active_version(family_dir):
    """The version patients get, or None before the first release."""
    path = family_dir / "active.json"
    return read_json(path)["version"] if path.is_file() else None


def set_active(family_dir, version):
    """Make version the one patients get."""
    if version not in versions(family_dir):
        raise PipelineError(f"version {version} isn't in the registry")
    write_json({"version": version}, family_dir / "active.json")


def active_metrics(family_dir):
    """The active version's metrics, for the release gate, or None before the first release."""
    version = active_version(family_dir)
    if version is None:
        return None
    return read_json(family_dir / f"v{version}" / "metrics.json")


def publish(bundle_dir, family_dir, version):
    """Copy the bundle in as v<version>, with its version and model_id in the manifest."""
    target = family_dir / f"v{version}"
    if target.exists():
        raise PipelineError(f"{target} already exists; a released version is never overwritten")
    shutil.copytree(bundle_dir, target)
    manifest = read_json(target / "manifest.json")
    manifest["version"] = version
    manifest["model_id"] = f"{manifest['family']}-v{version}"
    write_json(manifest, target / "manifest.json")
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
        self.family_dir = Path(cfg["resolved_paths"]["registry_dir"]) / cfg["export"]["family"]
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
