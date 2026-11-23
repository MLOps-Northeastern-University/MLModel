"""Run the whole model pipeline, as CI does.

    python -m scripts.run_model_pipeline
    python -m scripts.run_model_pipeline --config config/model.yaml
    python -m scripts.run_model_pipeline --no-mlflow     # without an MLflow server
    python -m scripts.run_model_pipeline --no-register   # every gate, but release nothing (CI on PRs)
    python -m scripts.run_model_pipeline --baseline      # only the hourly baseline: repeated CV and its report

Runs the steps in order, each logged to MLflow as a child of one model_pipeline run:

    train      every candidate cross-validated, refit and saved (P2)
    evaluate   intervals, plots, metrics.json and the validation gate (P1)
    bias       slices, the bias gate, and reweighted candidates for the ones that fail (P3)
    explain    which input groups drive each model (P3); reports only, never a gate
    select     the best candidate that passed every gate (P4)
    compare    the release gate against the active version (P4)
    export     the release bundle (P4)
    register   the bundle as the next version, made active (P4)

The comparison comes before export, so a model that can't replace the active one is never
packaged. Every step reads the data and the patient split itself (paths in config/model.yaml), so
the data must be there first: dvc pull in ../Data_Pipeline. Tuning isn't part of a run: run
scripts/steps/tune.py on its own, and train uses its settings when train.use_tuned is true.

A step that can't go on raises a core/errors.py error; the run then stops with exit code 1.
Alerts (scripts/alerts.py) go to the team when a gate fails, when the run crashes, and when
training finishes, saying whether a model was released.
"""

import argparse
import os
import platform
import sys
import traceback
import uuid

from scripts.alerts import notify
from scripts.core.errors import GateError, PipelineError
from scripts.data.folds import read_split
from scripts.data.loaders import load_dataset
from scripts.steps import baseline_cv
from scripts.steps.bias import CheckBias
from scripts.steps.compare_production import CompareProduction
from scripts.steps.evaluate import EvaluateModel
from scripts.steps.explain import ExplainModels
from scripts.steps.export import ExportModel
from scripts.steps.register import RegisterModel, active_metrics
from scripts.steps.select import SelectModel
from scripts.steps.train import TrainModels
from scripts.tracking.memory_tracker import MemoryTracker
from scripts.tracking.mlflow_tracker import MlflowTracker
from scripts.utils.config import load_config
from scripts.utils.logging_setup import configure_logging, get_logger, log_context, log_run_start, log_step

logger = get_logger("run_model_pipeline")

# Starts every alert's subject, to filter them in the inbox.
ALERT_PREFIX = "[MedNOVA model]"


# ---- what joins the steps ----------------------------------------------------------------

def training_stats(cfg, split, data=None):
    """The 7 hourly features' statistics over the development patients, for drift checks."""
    if data is None:
        data = load_dataset(cfg, groups=("features",), participants=split.development)
    else:
        data = data.subset(split.development)
    table = data.features.astype(float)
    return {"rows": len(table), "participants": len(split.development),
            "features": {name: {"mean": col.mean(), "std": col.std(), "min": col.min(),
                                "p50": col.median(), "max": col.max()}
                         for name, col in table.items()}}


# ---- the run -----------------------------------------------------------------------------

def run(cfg, tracker, release=True, data=None, split=None):
    """Run every step in order and return the released version.

    With release False, the run stops after export: every gate is checked and
    the bundle built, but nothing is registered, and it returns None.

    Args:
        data, split: A Dataset and patient split to use instead of reading them from
            the config's paths (tests pass small ones).

    Raises:
        PipelineError: A step couldn't go on, such as a failed gate (GateError).
    """
    register = RegisterModel(cfg, tracker)
    split = split or read_split(cfg)
    with tracker.run("model_pipeline", tags={"owner": "P5", "split_hash": split.hash}):
        with log_step(logger, "train", owner="P2"):
            candidates = TrainModels(cfg, tracker).run(data, split)
        with log_step(logger, "evaluate", owner="P1"):
            candidates = EvaluateModel(cfg, tracker).run(candidates)
        with log_step(logger, "check bias", owner="P3"):
            candidates = CheckBias(cfg, tracker).run(candidates, data, split)
        with log_step(logger, "explain", owner="P3"):
            ExplainModels(cfg, tracker).run(candidates, data, split)
        with log_step(logger, "select a model", owner="P4"):
            chosen = SelectModel(cfg, tracker).run(candidates)
        with log_step(logger, "compare with production", owner="P4"):
            CompareProduction(cfg, tracker).run(chosen["metrics"], active_metrics(register.family_dir))
        with log_step(logger, "export", owner="P4"):
            bundle_dir = ExportModel(cfg, tracker).run(chosen, training_stats(cfg, split, data))
        if not release:
            logger.info("Not registered: --no-register")
            return None
        with log_step(logger, "register", owner="P4"):
            return register.run(bundle_dir)


def run_baseline(cfg, tracker, data=None, split=None):
    """Only the hourly baseline: repeated cross-validation, the final model and its report
    (steps/baseline_cv.py). Nothing is selected, exported or released: it's there to compare with."""
    with log_step(logger, "baseline", owner="P1"):
        return baseline_cv.run(cfg, tracker, data, split)


def where():
    """Where the run happened, for alerts: the GitHub Actions run's page, or this machine."""
    run_id = os.environ.get("GITHUB_RUN_ID")
    if run_id:
        return (f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
                f"{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/{run_id}")
    return f"a local run on {platform.node()}"


def alert_failure(cfg, exc):
    """Tell the team the run stopped, and why."""
    if isinstance(exc, GateError):
        subject = f"{ALERT_PREFIX} The {exc.gate} gate failed"
    elif isinstance(exc, PipelineError):
        subject = f"{ALERT_PREFIX} Run stopped: {type(exc).__name__}"
    else:
        subject = f"{ALERT_PREFIX} Run crashed: {type(exc).__name__}"
    body = f"{exc}\n\nRun: {where()}"
    if not isinstance(exc, PipelineError):   # an unexpected error: the stack trace says where
        body += "\n\n" + "".join(traceback.format_exception(exc))
    notify(subject, body, cfg)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Parkinson's risk model pipeline")
    parser.add_argument("--config", default=None, help="Path to a config YAML")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Keep tracking in memory instead of logging to MLflow")
    parser.add_argument("--no-register", action="store_true",
                        help="Check every gate and build the bundle, but don't release it")
    parser.add_argument("--baseline", action="store_true",
                        help="Only the hourly baseline: repeated CV and a report in artifacts/baseline/")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.baseline:
        configure_logging(cfg)
        tracker = MemoryTracker() if args.no_mlflow else MlflowTracker.from_config(cfg)
        try:
            run_baseline(cfg, tracker)
        except PipelineError as exc:
            logger.error("Baseline stopped: %s", exc)
            return 1
        return 0
    configure_logging(cfg)
    # One id per run, on every log line, to find a run's lines in Cloud Logging.
    with log_context(run_id=uuid.uuid4().hex[:12]):
        log_run_start(logger, cfg)
        try:
            tracker = MemoryTracker() if args.no_mlflow else MlflowTracker.from_config(cfg)
            version = run(cfg, tracker, release=not args.no_register)
        except PipelineError as exc:
            logger.error("Model pipeline stopped: %s", exc)
            alert_failure(cfg, exc)
            return 1
        except Exception as exc:
            alert_failure(cfg, exc)
            raise
        logger.info("Model pipeline finished: %s", f"released v{version}" if version else "nothing released")
        if version:
            notify(f"{ALERT_PREFIX} {cfg['export']['family']} v{version} released",
                   f"Training finished, it passed every gate and is now the active version.\n\nRun: {where()}", cfg)
        else:
            notify(f"{ALERT_PREFIX} Training finished, every gate passed",
                   f"Nothing was released: this run only checks (--no-register).\n\nRun: {where()}", cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
