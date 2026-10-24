"""Run the whole pipeline without Airflow.

Used by `dvc repro` and for quick local runs:

    python -m scripts.run_pipeline --mode sample
    python -m scripts.run_pipeline --local-zips ~/Downloads   # use zips you already have
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

from scripts import pipeline_steps as steps
from scripts.alerts import notify
from scripts.utils.config import load_config
from scripts.utils.logging_setup import get_logger

logger = get_logger("run_pipeline")


def process_participants(cfg, participants):
    """Process every participant, collecting failures instead of stopping.

    Participants are independent, so up to run.parallel_participants of them
    run at the same time, each in its own process (the work is CPU-bound).
    Results keep the manifest's order either way.

    Returns:
        (list, list[dict]): Results of the participants that worked, and one
        {"participant_id", "error"} entry per participant that failed.
    """
    workers = min(cfg.get("run", {}).get("parallel_participants", 1), len(participants))
    if workers <= 1:
        outcomes = [_attempt(cfg, entry) for entry in participants]
    else:
        logger.info("Processing %d participants, %d at a time", len(participants), workers)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(_attempt, [cfg] * len(participants), participants))

    results, failures = [], []
    for entry, (result, error) in zip(participants, outcomes):
        if error is None:
            results.append(result)
        else:  # One bad participant should not stop the rest.
            logger.error("Skipping %s: %s", entry.get("name"), error)
            failures.append({"participant_id": Path(entry.get("name") or "?").stem, "error": error})
    return results, failures


def _attempt(cfg, entry):
    """Process one participant. Returns (result, None), or (None, error text) if it failed.

    Errors are returned, not raised, so a failure in a worker process reaches
    the parent as text, whatever the exception type.
    """
    try:
        return steps.step_participant(cfg, entry), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def report_failures(cfg, failures, total):
    """Save the failed participants, alert on them, and check the limit.

    Writes data/reports/failed_participants.json (empty list when none failed).

    Returns:
        bool: True when more than data.max_failed_fraction of participants failed.
    """
    # The same rule as the DAG's check_failures task.
    return steps.record_failures(cfg, failures, total, alert=notify)


def main(argv=None):
    parser = argparse.ArgumentParser(description="COPS data pipeline")
    parser.add_argument("--mode", choices=["sample", "full"], default=None)
    parser.add_argument("--config", default=None, help="Path to a config YAML")
    parser.add_argument("--local-zips", default=None, help="Folder with participant zips")
    args = parser.parse_args(argv)

    overrides = {"data": {"local_zip_dir": args.local_zips}} if args.local_zips else None
    cfg = load_config(args.config, overrides)
    run_id = datetime.now().strftime("cli_%Y%m%dT%H%M%S")

    manifest = steps.step_manifest(cfg, args.mode)
    demographics = steps.step_demographics(cfg, manifest)
    results, failures = process_participants(cfg, manifest["participants"])
    if report_failures(cfg, failures, len(manifest["participants"])) or not results:
        logger.error("Stopping: too many participants failed (see failed_participants.json)")
        return 1

    merged = steps.step_merge(cfg, results, demographics)
    features = steps.step_clean(cfg, merged)
    steps.step_validate(cfg, features, run_id)
    steps.step_anomalies(cfg, features)
    report = steps.step_bias(cfg, features)
    logger.info("Pipeline finished. Features: %s | Bias report: %s", features, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
