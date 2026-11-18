"""Load the pipeline configuration and resolve project paths."""

import copy
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "model.yaml"


def load_config(path=None, overrides=None):
    """Read the YAML config and return it as a dict.

    Args:
        path: Optional path to a config file. Defaults to the MODEL_CONFIG
            environment variable, then config/model.yaml.
        overrides: Optional nested dict merged on top of the file values.

    Returns:
        dict: The configuration, with paths resolved to absolute Paths.
    """
    config_path = Path(path or os.environ.get("MODEL_CONFIG", DEFAULT_CONFIG_PATH))
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
    # Create the folders this pipeline writes to. Input files, like the
    # data pipeline's table, are only read.
    for key, value in cfg["resolved_paths"].items():
        if key.endswith("_dir"):
            value.mkdir(parents=True, exist_ok=True)
        elif key == "log_file":
            value.parent.mkdir(parents=True, exist_ok=True)
    return cfg


def choose_candidates(cfg, names):
    """A copy of cfg that trains only the candidates named, in config order.

    For a run that retrains some models, not all (run_model_pipeline --candidates). The steps all
    build their candidates from cfg, so they follow:

    - models.candidates keeps only these;
    - tune.candidates keeps only these, so --tune tunes only what this run trains;
    - train.leakage_check.candidates keeps the ones among these. If it named none of them, the check
      runs on the first candidate named here instead, so every run still has it. It's skipped only
      when the config turns it off (an empty list).

    bias.mitigation.candidates and explain.candidates need nothing: they only ever match a candidate
    the run trained.

    Raises:
        ValueError: No names, a repeated name, or a name that isn't a candidate in the config.
    """
    names = list(names)
    known = [c.get("name") for c in cfg["models"]["candidates"]]
    unknown = sorted(set(names) - set(known))
    if unknown:
        raise ValueError(f"unknown candidate(s) {unknown}; the config has {known}")
    if not names or len(set(names)) != len(names):
        raise ValueError(f"name each candidate to train once, from {known}; got {names}")

    out = copy.deepcopy(cfg)
    out["models"]["candidates"] = [c for c in out["models"]["candidates"] if c["name"] in names]
    chosen = [c["name"] for c in out["models"]["candidates"]]      # in config order

    tune = out.get("tune") or {}
    if tune.get("candidates"):
        tune["candidates"] = {name: c for name, c in tune["candidates"].items() if name in chosen}

    check = out.get("train", {}).get("leakage_check") or {}
    if check.get("candidates"):
        kept = [name for name in check["candidates"] if name in chosen]
        check["candidates"] = kept or chosen[:1]
    return out


def _deep_merge(base, extra):
    """Return a copy of base with extra merged in, recursively."""
    merged = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
