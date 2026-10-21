"""Load the pipeline configuration and resolve project paths."""

import copy
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"


def load_config(path=None, overrides=None):
    """Read the YAML config and return it as a dict.

    Args:
        path: Optional path to a config file. Defaults to the COPS_CONFIG
            environment variable, then config/pipeline.yaml.
        overrides: Optional nested dict merged on top of the file values.

    Returns:
        dict: The configuration, with paths resolved to absolute Paths.
    """
    config_path = Path(path or os.environ.get("COPS_CONFIG", DEFAULT_CONFIG_PATH))
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if overrides:
        cfg = _deep_merge(cfg, overrides)

    root = Path(cfg.get("project_root", PROJECT_ROOT))
    cfg["resolved_paths"] = {
        key: (Path(value) if Path(value).is_absolute() else root / value)
        for key, value in cfg["paths"].items()
    }
    for key, value in cfg["resolved_paths"].items():
        target = value.parent if key == "log_file" else value
        target.mkdir(parents=True, exist_ok=True)
    return cfg


def _deep_merge(base, extra):
    """Return a copy of base with extra merged in, recursively."""
    merged = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
