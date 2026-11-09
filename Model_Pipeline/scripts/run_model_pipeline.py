"""Run the whole model pipeline, as CI does.

    python -m scripts.run_model_pipeline
    python -m scripts.run_model_pipeline --config config/model.yaml

A placeholder for now: it lists the steps in order. Each owner's step is
called from here once it's merged (see "Who builds what" in the README).
"""

import argparse
import sys

from scripts.utils.config import load_config
from scripts.utils.logging_setup import get_logger

logger = get_logger("run_model_pipeline")

# (step, owner), in the order they run
STEPS = [
    ("load data", "P1"),
    ("train and tune", "P2"),
    ("validate", "P1"),
    ("check bias", "P3"),
    ("select a model", "P4"),
    ("export to TFLite", "P4"),
    ("compare with production", "P4"),
    ("push to the registry", "P4"),
]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Parkinson's risk model pipeline")
    parser.add_argument("--config", default=None, help="Path to a config YAML")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    logger.info("Training data: %s", cfg["resolved_paths"]["features_csv"])
    for name, owner in STEPS:
        logger.info("Step '%s' (%s): not built yet", name, owner)
    return 0


if __name__ == "__main__":
    sys.exit(main())
