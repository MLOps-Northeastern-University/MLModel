"""Reads config/retrain.yaml, the file a PR changes to ask CI to train the model.

    python -m scripts.retrain_request check                         # is retrain.yaml valid? (every PR)
    python -m scripts.retrain_request conf --before old_retrain.yaml  # the DAG's config, if asked

check fails (exit 1) when the file is malformed or names a candidate model.yaml doesn't have, so
the PR that breaks it fails its checks, not the merge after it.

conf is what CI runs on a merge to main, with --before the file as it was in the commit before
(git show HEAD^:Model_Pipeline/config/retrain.yaml). When request changed, it prints the model
DAG's trigger config as JSON:

    {"candidates": ["motor_causal_transformer"], "tune": true, "reason": "...", "release": true}

When request didn't change (or there's no file before, as on the commit that adds it), it prints
nothing, and CI trains nothing.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from scripts.utils.config import PROJECT_ROOT, load_config

RETRAIN_FILE = PROJECT_ROOT / "config" / "retrain.yaml"
KEYS = {"request", "reason", "candidates", "tune"}


class RetrainRequestError(ValueError):
    """retrain.yaml is malformed, or names a candidate the config doesn't have."""


def read(text, known_candidates):
    """The request in retrain.yaml's text, checked. Returns {request, reason, candidates, tune}."""
    request = yaml.safe_load(text) or {}
    if not isinstance(request, dict):
        raise RetrainRequestError("retrain.yaml must be a mapping of request, reason, candidates and tune")
    unknown_keys = set(request) - KEYS
    if unknown_keys:
        raise RetrainRequestError(f"retrain.yaml has unknown key(s) {sorted(unknown_keys)}; allowed: {sorted(KEYS)}")
    number = request.get("request")
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise RetrainRequestError(f"request must be a whole number of 1 or more, not {number!r}")
    reason = request.get("reason") or ""
    if not isinstance(reason, str):
        raise RetrainRequestError(f"reason must be text, not {reason!r}")
    candidates = request.get("candidates") or []
    if not isinstance(candidates, list) or not all(isinstance(c, str) for c in candidates):
        raise RetrainRequestError(f"candidates must be a list of names, not {candidates!r}")
    unknown = sorted(set(candidates) - set(known_candidates))
    if unknown:
        raise RetrainRequestError(f"candidates names {unknown}, not in models.candidates {list(known_candidates)}")
    tune = request.get("tune", False)
    if not isinstance(tune, bool):
        raise RetrainRequestError(f"tune must be true or false, not {tune!r}")
    return {"request": number, "reason": reason, "candidates": candidates, "tune": tune}


def dag_conf(now, before):
    """The model DAG's trigger config when request changed since before, else None."""
    if before is None or now["request"] == before["request"]:
        return None
    return {"candidates": now["candidates"], "tune": now["tune"], "reason": now["reason"], "release": True}


def known_candidates(cfg):
    return [c["name"] for c in cfg["models"]["candidates"]]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read config/retrain.yaml (see scripts/retrain_request.py)")
    parser.add_argument("command", choices=["check", "conf"])
    parser.add_argument("--file", default=str(RETRAIN_FILE), help="the retrain.yaml to read")
    parser.add_argument("--before", default=None, help="conf only: the file as it was in the commit before")
    parser.add_argument("--config", default=None, help="Path to a config YAML")
    args = parser.parse_args(argv)
    names = known_candidates(load_config(args.config))
    try:
        now = read(Path(args.file).read_text(encoding="utf-8"), names)
        if args.command == "check":
            print(f"retrain.yaml is valid: request {now['request']}")
            return 0
        before_path = Path(args.before) if args.before else None
        before = None
        if before_path and before_path.exists() and before_path.read_text(encoding="utf-8").strip():
            # The old file is read leniently: a candidate removed since then mustn't fail this merge.
            before = {"request": (yaml.safe_load(before_path.read_text(encoding="utf-8")) or {}).get("request")}
        conf = dag_conf(now, before)
        if conf is not None:
            print(json.dumps(conf))
        return 0
    except RetrainRequestError as exc:
        print(f"retrain.yaml: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
