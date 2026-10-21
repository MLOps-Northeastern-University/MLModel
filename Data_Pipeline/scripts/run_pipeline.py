"""Run the whole pipeline without Airflow.

Used by `dvc repro` and for quick local runs:

    python -m scripts.run_pipeline --mode sample
    python -m scripts.run_pipeline --local-zips ~/Downloads   # use zips you already have
"""

import argparse
import sys
from datetime import datetime

from scripts import pipeline_steps as steps
from scripts.utils.config import load_config
from scripts.utils.logging_setup import get_logger

logger = get_logger("run_pipeline")


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
    results = []
    for entry in manifest["participants"]:
        try:
            results.append(steps.step_participant(cfg, entry))
        except Exception as exc:  # One bad participant should not stop the rest.
            logger.error("Skipping %s: %s", entry.get("name"), exc)
    if not results:
        logger.error("No participants processed")
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
