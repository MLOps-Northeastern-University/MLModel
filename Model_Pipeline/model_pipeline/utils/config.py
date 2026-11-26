"""Load the pipeline configuration and resolve project paths."""

import copy
import os
from pathlib import Path

import yaml

# Model_Pipeline/, two folders up from this file. Relative paths in model.yaml are joined to it (or to the
# config's project_root, which tests set to a temporary folder), whatever folder a command runs from.
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

    # Overrides are merged before the paths are resolved, so overriding project_root or a path (the tests
    # point both at tmp_path) moves every output with it.
    if overrides:
        cfg = _deep_merge(cfg, overrides)

    # cfg["paths"] keeps the paths as written and the code uses resolved_paths. log_run_start leaves
    # resolved_paths out of the config hash, since the same config resolves differently on each machine.
    root = Path(cfg.get("project_root", PROJECT_ROOT))
    cfg["resolved_paths"] = {
        key: (Path(value) if Path(value).is_absolute() else root / value)
        for key, value in cfg["paths"].items()
    }
    # Create the folders this pipeline writes to. Input files, like the
    # data pipeline's table, are only read.
    # windows_folder is an input too and doesn't end in _dir, so it isn't created empty before dvc pull has
    # filled it.
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
    # Unknown names are reported first, so a typo gets the list of real names. Both errors are ValueErrors,
    # which run_model_pipeline turns into a usage error and run_step start into exit code 2.
    unknown = sorted(set(names) - set(known))
    if unknown:
        raise ValueError(f"unknown candidate(s) {unknown}; the config has {known}")
    if not names or len(set(names)) != len(names):
        raise ValueError(f"name each candidate to train once, from {known}; got {names}")

    out = copy.deepcopy(cfg)
    out["models"]["candidates"] = [c for c in out["models"]["candidates"] if c["name"] in names]
    chosen = [c["name"] for c in out["models"]["candidates"]]      # in config order

    # tune and check are the dicts inside out, not copies, so assigning their candidates changes out. A
    # missing section falls back to a throwaway {}, and an empty list (a check turned off) is falsy, so both
    # are left as they are.
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
    # Only dicts merge key by key; a list or any other value replaces the base's. So an override can add or
    # change keys but never remove one, which is why tests/pipeline_run.py assigns validation_gate.min after
    # loading instead.
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
