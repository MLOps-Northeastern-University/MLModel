"""Run one step of the model pipeline on its own, as each task of the Airflow DAG does.

    python -m scripts.run_step start --candidates window_logistic --reason "new data"
    python -m scripts.run_step train
    python -m scripts.run_step evaluate
    ...
    python -m scripts.run_step register
    python -m scripts.run_step finish

start opens the run: it saves the run's choices to artifacts/run.json and opens the model_pipeline
run in MLflow. Every step then reads run.json, so its command needs no options, and logs under that
MLflow run, so MLflow shows one run with a child per step, as with run_model_pipeline. finish
closes it: FINISHED if every step passed, FAILED otherwise, and tells the team the result.

start takes the same options as run_model_pipeline (--candidates, --tune, --reason, --no-register,
--config). The steps are the ones in pipeline_steps.py, which also says what each reads and writes.

Exit codes: 0 when the step passed, 1 when it stopped (a failed gate, missing input), so Airflow marks
the task failed and skips the ones after it. A failed step alerts the team, as run_model_pipeline does.
"""

import argparse
import sys
import uuid
from pathlib import Path

from scripts.alerts import notify
from scripts.core.errors import PipelineError
from scripts.core.tracker import RunStatus
from scripts.data.folds import read_split
from scripts.pipeline_steps import STEPS
from scripts.run_model_pipeline import ALERT_PREFIX, alert_failure, run_tags, where, with_tuning
from scripts.tracking.mlflow_tracker import MlflowTracker
from scripts.utils.config import choose_candidates, load_config
from scripts.utils.file_io import read_json, write_json
from scripts.utils.logging_setup import configure_logging, get_logger, log_context, log_step

logger = get_logger("run_step")

RUN_FILE = "run.json"


# ---- the run's choices, saved by start ------------------------------------------------------

def run_file(cfg):
    return Path(cfg["resolved_paths"]["artifacts_dir"]) / RUN_FILE


def read_run(cfg):
    path = run_file(cfg)
    if not path.exists():
        raise PipelineError(f"no {path}: run `python -m scripts.run_step start` first")
    return read_json(path)


def config_for(run, config_path=None):
    """The config the run's steps use: only its candidates, and tuned settings if it tunes."""
    cfg = load_config(config_path or run.get("config"))
    if run["candidates"]:
        cfg = choose_candidates(cfg, run["candidates"])
    if run["tune"]:
        cfg = with_tuning(cfg)
    return cfg


# ---- the commands ------------------------------------------------------------------------

def start(args, tracker_factory):
    """Save the run's choices and open its MLflow run. Prints the run's MLflow id."""
    choices = {"candidates": args.candidates or [], "tune": args.tune, "reason": args.reason,
               "release": not args.no_register, "config": args.config}
    try:
        cfg = config_for(choices)
    except ValueError as exc:                       # a candidate the config doesn't have
        logger.error("Can't start: %s", exc)
        return 2
    tracker = tracker_factory(cfg)
    run_id = tracker.open_run("model_pipeline",
                              tags=run_tags(cfg, read_split(cfg), args.tune, args.reason))
    write_json({**choices, "mlflow_run_id": run_id}, run_file(cfg))
    print(run_id)
    return 0


def step(name, args, tracker_factory):
    """Run one step under the run's MLflow run."""
    try:
        run = read_run(load_config(args.config))
    except PipelineError as exc:
        logger.error("Step %s can't run: %s", name, exc)
        return 1
    cfg = config_for(run, args.config)
    if name == "register" and not run["release"]:
        logger.info("Not registered: the run was started with --no-register")
        return 0
    if name == "tune" and not run["tune"]:
        logger.info("Not tuned: the run was started without --tune")
        return 0
    function, owner = STEPS[name]
    tracker = tracker_factory(cfg)
    try:
        with tracker.attach(run["mlflow_run_id"]), log_step(logger, name, owner=owner):
            result = function(cfg, tracker)
    except PipelineError as exc:
        logger.error("Step %s stopped: %s", name, exc)
        alert_failure(cfg, exc)
        return 1
    except Exception as exc:
        alert_failure(cfg, exc)
        raise
    if name == "register":
        write_json({**run, "released_version": result}, run_file(cfg))
    return 0


def finish(args, tracker_factory):
    """Close the run's MLflow run, and tell the team the result when it passed."""
    run = read_run(load_config(args.config))
    cfg = config_for(run, args.config)
    status = RunStatus.FINISHED if args.status == "finished" else RunStatus.FAILED
    tracker_factory(cfg).close_run(run["mlflow_run_id"], status)
    if status is RunStatus.FAILED:
        return 0                                    # the step that failed already told the team why
    version = run.get("released_version")
    if version:
        notify(f"{ALERT_PREFIX} {cfg['export']['family']} v{version} released",
               f"Training finished, it passed every gate and is now the active version.\n\nRun: {where()}", cfg)
    else:
        notify(f"{ALERT_PREFIX} Training finished, every gate passed",
               f"Nothing was released: this run only checks (--no-register).\n\nRun: {where()}", cfg)
    return 0


# ---- the command line --------------------------------------------------------------------

def parser():
    p = argparse.ArgumentParser(description="Run one step of the model pipeline (see scripts/run_step.py)")
    p.add_argument("command", choices=["start", *STEPS, "finish"])
    p.add_argument("--config", default=None, help="Path to a config YAML")
    start_options = p.add_argument_group("start only")
    start_options.add_argument("--candidates", nargs="+", metavar="NAME", default=None,
                               help="Train only these candidates (names from models.candidates); default: all")
    start_options.add_argument("--tune", action="store_true", help="Tune the candidates before training")
    start_options.add_argument("--reason", default=None, help="Why this run happened, tagged on the MLflow run")
    start_options.add_argument("--no-register", action="store_true", help="Check every gate, but release nothing")
    p.add_argument("--status", choices=["finished", "failed"], default="finished",
                   help="finish only: how the run ended")
    return p


def main(argv=None, tracker_factory=MlflowTracker.from_config):
    args = parser().parse_args(argv)
    configure_logging(load_config(args.config))
    with log_context(run_id=uuid.uuid4().hex[:12], step=args.command):
        if args.command == "start":
            return start(args, tracker_factory)
        if args.command == "finish":
            return finish(args, tracker_factory)
        return step(args.command, args, tracker_factory)


if __name__ == "__main__":
    sys.exit(main())
