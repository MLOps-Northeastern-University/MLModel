"""Run the whole model pipeline, as CI does.

    python -m scripts.run_model_pipeline
    python -m scripts.run_model_pipeline --config config/model.yaml
    python -m scripts.run_model_pipeline --no-mlflow     # without an MLflow server
    python -m scripts.run_model_pipeline --no-register   # every gate, but release nothing (CI on PRs)

Runs the steps in order: load, train and tune, evaluate, bias, select, compare
with production, export, register. The comparison comes before export, so a
model that can't replace the active one is never packaged.

Load, train, evaluate and bias aren't built yet (P1, P2, P3), so stand-ins make
outputs in the shape the next step expects: one candidate with placeholder
metrics and a fake model file. Each stand-in is replaced by the real step once
its owner merges it. A step that can't go on raises a core/errors.py error; the
run then stops with exit code 1.
"""

import argparse
import sys
import uuid

from scripts.core.errors import PipelineError
from scripts.steps.compare_production import CompareProduction
from scripts.steps.export import ExportModel
from scripts.steps.register import RegisterModel, active_metrics
from scripts.steps.select import SelectModel
from scripts.tracking.memory_tracker import MemoryTracker
from scripts.tracking.mlflow_tracker import MlflowTracker
from scripts.utils.config import load_config
from scripts.utils.logging_setup import configure_logging, get_logger, log_context, log_run_start, log_step

logger = get_logger("run_model_pipeline")

# Every stand-in metric gets this score, so the stand-in candidate passes select.
STAND_IN_SCORE = 0.5


# ---- stand-ins, until P1, P2 and P3 merge their steps ------------------------------------

def stand_in_load(cfg):
    """PLACEHOLDER (P1, data/loaders.py): no data is read yet.

    Returns:
        (None, dict): The dataset, and its training statistics for the bundle.
    """
    logger.warning("Stand-in: no data read from %s", cfg["resolved_paths"]["features_csv"])
    return None, {"rows": 0, "features": {}}


def stand_in_train(cfg, dataset):
    """PLACEHOLDER (P2, steps/train.py and tune.py): one candidate and a fake model file."""
    model_path = cfg["resolved_paths"]["artifacts_dir"] / "stand_in" / "model.bin"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"stand-in model")
    metrics = {rule["metric"]: STAND_IN_SCORE for rule in cfg["release_gate"]["rules"]}
    metrics[cfg["selection"]["metric"]] = STAND_IN_SCORE
    logger.warning("Stand-in: one candidate with every metric at %.1f", STAND_IN_SCORE)
    return [{"name": "stand-in", "metrics": metrics, "model_path": model_path,
             "gates_passed": False, "exportable": True}]


def stand_in_evaluate(cfg, dataset, candidates):
    """PLACEHOLDER (P1, steps/evaluate.py): every candidate passes the validation gate."""
    logger.warning("Stand-in: validation gate not checked")
    return [{**c, "gates_passed": True} for c in candidates]


def stand_in_bias(cfg, dataset, candidates):
    """PLACEHOLDER (P3, steps/bias.py): every candidate passes the bias gate."""
    logger.warning("Stand-in: bias gate not checked")
    return candidates


# ---- the run -----------------------------------------------------------------------------

def run(cfg, tracker, release=True):
    """Run every step in order and return the released version.

    With release False, the run stops after export: every gate is checked and
    the bundle built, but nothing is registered, and it returns None.

    Raises:
        PipelineError: A step couldn't go on, such as a failed gate (GateError).
    """
    register = RegisterModel(cfg, tracker)
    with tracker.run("model_pipeline", tags={"owner": "P5"}):
        with log_step(logger, "load data", owner="P1"):
            dataset, training_stats = stand_in_load(cfg)
        with log_step(logger, "train and tune", owner="P2"):
            candidates = stand_in_train(cfg, dataset)
        with log_step(logger, "evaluate", owner="P1"):
            candidates = stand_in_evaluate(cfg, dataset, candidates)
        with log_step(logger, "check bias", owner="P3"):
            candidates = stand_in_bias(cfg, dataset, candidates)
        with log_step(logger, "select a model", owner="P4"):
            chosen = SelectModel(cfg, tracker).run(candidates)
        with log_step(logger, "compare with production", owner="P4"):
            CompareProduction(cfg, tracker).run(chosen["metrics"], active_metrics(register.family_dir))
        with log_step(logger, "export", owner="P4"):
            bundle_dir = ExportModel(cfg, tracker).run(chosen, chosen["model_path"], training_stats)
        if not release:
            logger.info("Not registered: --no-register")
            return None
        with log_step(logger, "register", owner="P4"):
            return register.run(bundle_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Parkinson's risk model pipeline")
    parser.add_argument("--config", default=None, help="Path to a config YAML")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Keep tracking in memory instead of logging to MLflow")
    parser.add_argument("--no-register", action="store_true",
                        help="Check every gate and build the bundle, but don't release it")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    configure_logging(cfg)
    # One id per run, on every log line, to find a run's lines in Cloud Logging.
    with log_context(run_id=uuid.uuid4().hex[:12]):
        log_run_start(logger, cfg)
        try:
            tracker = MemoryTracker() if args.no_mlflow else MlflowTracker.from_config(cfg)
            version = run(cfg, tracker, release=not args.no_register)
        except PipelineError as exc:
            logger.error("Model pipeline stopped: %s", exc)
            return 1
        logger.info("Model pipeline finished: %s", f"released v{version}" if version else "nothing released")
    return 0


if __name__ == "__main__":
    sys.exit(main())
