"""Run the whole model pipeline, as CI does.

    python -m model_pipeline.run_model_pipeline
    python -m model_pipeline.run_model_pipeline --config config/model.yaml
    python -m model_pipeline.run_model_pipeline --no-mlflow     # without an MLflow server
    python -m model_pipeline.run_model_pipeline --no-register   # every gate, but release nothing (CI on PRs)
    python -m model_pipeline.run_model_pipeline --candidates window_logistic        # retrain only these
    python -m model_pipeline.run_model_pipeline --candidates motor_causal_transformer --tune --reason "deeper encoder"

Runs the steps in order, each logged to MLflow as a child of one model_pipeline run:

    tune       only with --tune: the candidates' settings searched again (P2)
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
the data must be there first: dvc pull in ../Data_Pipeline. The steps are in pipeline_steps.py and
hand over through files in artifacts/, so the Airflow DAG can run each one on its own (run_step.py).

Which candidates and whether to tune are chosen per run, not in the config, so a run trains only
what changed:

    --candidates   the candidates to train (names from models.candidates); all of them when left out.
                   The others aren't trained, and the release gate compares the best of these with
                   the active version, whatever its family.
    --tune         search the candidates' settings again first (tune.candidates), and train with them.
                   Without it, train reuses the last tuned settings (train.use_tuned), even if they
                   were tuned on older data; see experiments.read_tuned.
    --reason       why this run happened (a change, drift), kept as a tag on the MLflow run.

A step that can't go on raises a core/errors.py error; the run then stops with exit code 1.
Alerts (model_pipeline/alerts.py) go to the team when a gate fails, when the run crashes, and when
training finishes, saying whether a model was released.
"""

import argparse
import copy
import os
import platform
import sys
import traceback
import uuid

from model_pipeline.alerts import notify
from model_pipeline.core.errors import GateError, PipelineError
from model_pipeline.data.folds import read_split
from model_pipeline.pipeline_steps import STEPS, steps_for
from model_pipeline.tracking.memory_tracker import MemoryTracker
from model_pipeline.tracking.mlflow_tracker import MlflowTracker
from model_pipeline.utils.config import choose_candidates, load_config
from model_pipeline.utils.logging_setup import configure_logging, get_logger, log_context, log_run_start, log_step

logger = get_logger("run_model_pipeline")

# Starts every alert's subject, to filter them in the inbox.
# run_step.py imports ALERT_PREFIX, alert_failure, where, run_tags and with_tuning from this module, so a
# run made of Airflow tasks sends the same alerts and tags its MLflow run the same way as a whole run.
ALERT_PREFIX = "[MedNOVA model]"


# ---- the run -----------------------------------------------------------------------------

def run(cfg, tracker, release=True, data=None, split=None, tune=False, reason=None):
    """Run every step in order and return the released version.

    With release False, the run stops after export: every gate is checked and
    the bundle built, but nothing is registered, and it returns None.

    Args:
        cfg: The config, with only the candidates to train (utils/config.choose_candidates).
        data, split: A Dataset and patient split to use instead of reading them from
            the config's paths (tests pass small ones).
        tune: Search the candidates' settings first, and train with them.
        reason: Why this run happened, for the MLflow run's tags.

    Raises:
        PipelineError: A step couldn't go on, such as a failed gate (GateError).
    """
    # Tuning has to change the config before any step sees it: train reads train.use_tuned to decide whether
    # to use what tune writes to artifacts/tuned_params.json. with_tuning works on a copy, so the config the
    # caller passed in is left as it was.
    if tune:
        cfg = with_tuning(cfg)
    # The split is read once and handed to every step, so the split_hash tag and every step's folds come from
    # the same file. data stays None in a real run, and each step loads the data it needs from the config's paths.
    split = split or read_split(cfg)
    results = {}
    # Each step opens its own tracker run inside this one, so MLflow shows one model_pipeline run with a child
    # per step. A step that raises ends this run FAILED, and the error goes on to the caller (main alerts).
    with tracker.run("model_pipeline", tags=run_tags(cfg, split, tune, reason)):
        for name in steps_for(tune=tune, release=release):
            step, owner = STEPS[name]
            # log_step logs the step's start, end and duration, with the owner on every line in between, and a failure
            # with its stack trace, so the logs show which step stopped a run and whose code it was.
            with log_step(logger, name, owner=owner):
                results[name] = step(cfg, tracker, data, split)
    if not release:
        logger.info("Not registered: --no-register")
        return None
    return results["register"]                                  # the released version


def with_tuning(cfg):
    """A copy of cfg that trains with what tuning finds, even if the config doesn't use tuned settings."""
    cfg = copy.deepcopy(cfg)
    cfg["train"]["use_tuned"] = True
    return cfg


def run_tags(cfg, split, tune, reason):
    """The model_pipeline run's tags: what it trained, on which split, whether it tuned, and why."""
    tags = {"owner": "P5", "split_hash": split.hash,
            "candidates": ",".join(c["name"] for c in cfg["models"]["candidates"]),
            "tuned_this_run": str(tune).lower()}
    if reason:
        tags["reason"] = reason
    return tags


def where():
    """Where the run happened, for alerts: the GitHub Actions run's page, or this machine."""
    # GitHub Actions sets GITHUB_RUN_ID, GITHUB_SERVER_URL and GITHUB_REPOSITORY in every job, which is
    # enough to link straight to the run's page. Anywhere else the alert names the machine instead.
    run_id = os.environ.get("GITHUB_RUN_ID")
    if run_id:
        return (f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
                f"{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/{run_id}")
    return f"a local run on {platform.node()}"


def alert_failure(cfg, exc):
    """Tell the team the run stopped, and why."""
    # GateError is checked first because it is also a PipelineError. A gate, or any other PipelineError, is a
    # stop the pipeline makes on purpose and its message says why; anything else is a bug, so the body gets
    # the stack trace too.
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
    """Parse the command line, run the pipeline, and alert the team with the outcome.

    Args:
        argv: The arguments, without the program name. Defaults to sys.argv[1:].

    Returns:
        int: 0 when every step passed, whether or not a version was released, and 1 when a step
        stopped the run with a PipelineError (a failed gate, a missing input).

    Raises:
        SystemExit: Bad arguments, such as a candidate the config doesn't have (argparse's exit code 2).
        Exception: Any error that isn't a PipelineError, after alerting, so CI shows its traceback.
    """
    parser = argparse.ArgumentParser(description="Parkinson's risk model pipeline")
    parser.add_argument("--config", default=None, help="Path to a config YAML")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Keep tracking in memory instead of logging to MLflow")
    parser.add_argument("--no-register", action="store_true",
                        help="Check every gate and build the bundle, but don't release it")
    parser.add_argument("--candidates", nargs="+", metavar="NAME", default=None,
                        help="Train only these candidates (names from models.candidates); default: all")
    parser.add_argument("--tune", action="store_true",
                        help="Search the candidates' settings again before training (slow)")
    parser.add_argument("--reason", default=None, help="Why this run happened, tagged on the MLflow run")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    # A candidate name the config doesn't have is reported as a usage error (exit code 2) before MLflow or
    # any step starts, so a typo in --candidates costs nothing.
    if args.candidates:
        try:
            cfg = choose_candidates(cfg, args.candidates)
        except ValueError as exc:
            parser.error(str(exc))
    configure_logging(cfg)
    # One id per run, on every log line, to find a run's lines in Cloud Logging.
    with log_context(run_id=uuid.uuid4().hex[:12]):
        log_run_start(logger, cfg)
        # Building the tracker is inside the try too, so failing to reach MLflow is alerted like any other error.
        # A PipelineError is an expected stop: logged, alerted, and exit code 1 fails the CI job. Anything else
        # is alerted and re-raised, so its traceback reaches the CI log as well.
        try:
            tracker = MemoryTracker() if args.no_mlflow else MlflowTracker.from_config(cfg)
            version = run(cfg, tracker, release=not args.no_register, tune=args.tune, reason=args.reason)
        except PipelineError as exc:
            logger.error("Model pipeline stopped: %s", exc)
            alert_failure(cfg, exc)
            return 1
        except Exception as exc:
            alert_failure(cfg, exc)
            raise
        logger.info("Model pipeline finished: %s", f"released v{version}" if version else "nothing released")
        # A run that gets here sends one of these two; a run that stopped sent its alert above instead.
        if version:
            notify(f"{ALERT_PREFIX} {cfg['export']['family']} v{version} released",
                   f"Training finished, it passed every gate and is now the active version.\n\nRun: {where()}", cfg)
        else:
            notify(f"{ALERT_PREFIX} Training finished, every gate passed",
                   f"Nothing was released: this run only checks (--no-register).\n\nRun: {where()}", cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
