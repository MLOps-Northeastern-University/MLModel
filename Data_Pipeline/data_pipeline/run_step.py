"""Run one step of the data pipeline on its own, as each task of the Airflow DAG does.

    python -m data_pipeline.run_step manifest --mode sample   # prints the participants, one JSON list
    python -m data_pipeline.run_step patients
    python -m data_pipeline.run_step participant COPS-12.zip  # one per participant, in parallel
    python -m data_pipeline.run_step check_failures
    python -m data_pipeline.run_step build
    python -m data_pipeline.run_step checks
    python -m data_pipeline.run_step notify_success

The same functions as `python -m data_pipeline.pipeline` (pipeline.py), which runs them all in one process.
Here each runs in its own process (the DAG runs each in the data pipeline's container), so the steps
hand over only through files:

    step             reads                                  writes
    manifest         the OSF listing, or data/raw/          data/raw/manifest.json
    patients         manifest.json, Demographics.csv        data/interim/patients.csv
    participant      manifest.json, patients.csv, its zip   data/interim/<pid>/
    check_failures   manifest.json, data/interim/<pid>/     data/reports/failed_participants.json
    build            data/interim/                          data/processed/
    checks           data/processed/                        data/reports/checks.json, bias_report

manifest also clears the last run's data/interim/<pid>/ folders, so a participant that fails in this
run can't be counted from what an earlier run left behind.

Exit codes: 0 when the step passed, 1 when it stopped (too many participants failed, the dataset
failed its checks, a download that kept failing), so Airflow marks the task failed.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from data_pipeline import acquire, checks, pipeline
from data_pipeline.utils import get_logger, load_config, notify

logger = get_logger("run_step")

PATIENTS_FILE = "patients.csv"


# ---- what the steps hand over ------------------------------------------------------------

def patients_path(cfg):
    return cfg["resolved_paths"]["interim_dir"] / PATIENTS_FILE


def participant_folder(cfg, entry):
    return cfg["resolved_paths"]["interim_dir"] / Path(entry["name"]).stem


def finished_participants(cfg, manifest):
    """The participants of this run whose outputs are complete, as participant_step returns them."""
    results = []
    for entry in manifest["participants"]:
        folder = participant_folder(cfg, entry)
        if (folder / "meta.json").exists():                 # written last, once everything else is
            results.append({"participant_id": folder.name, "dir": str(folder)})
    return results


# ---- the steps ---------------------------------------------------------------------------

def manifest(cfg, mode):
    """Choose the participants, clear what the last run left of them, and print their zip names."""
    chosen = acquire.fetch_manifest(cfg, mode)
    for entry in chosen["participants"]:
        shutil.rmtree(participant_folder(cfg, entry), ignore_errors=True)
    # The last line of output: the DAG reads it to start one task per participant.
    print(json.dumps([entry["name"] for entry in chosen["participants"]]))


def patients(cfg):
    path = pipeline.patients_step(cfg, acquire.load_manifest(cfg))
    logger.info("Patients table: %s", path)


def participant(cfg, name):
    entries = {entry["name"]: entry for entry in acquire.load_manifest(cfg)["participants"]}
    if name not in entries:
        raise acquire.AcquisitionError(f"{name} isn't in this run's manifest")
    pipeline.participant_step(cfg, entries[name], patients_path(cfg))


def check_failures(cfg):
    chosen = acquire.load_manifest(cfg)
    pipeline.check_failures(cfg, chosen, finished_participants(cfg, chosen))


def build(cfg):
    chosen = acquire.load_manifest(cfg)
    pipeline.build(cfg, finished_participants(cfg, chosen), patients_path(cfg), chosen["mode"])


def run_checks(cfg):
    checks.run_checks(cfg)


def notify_success(cfg):
    notify("[COPS pipeline] Run succeeded", f"Dataset: {cfg['resolved_paths']['processed_dir']}", cfg)


STEPS = ["manifest", "patients", "participant", "check_failures", "build", "checks", "notify_success"]

# What stops a step on purpose: exit 1, with the reason in the log, rather than a stack trace.
STOPS = (acquire.AcquisitionError, pipeline.TooManyFailures, checks.DataCheckError)


# ---- the command line --------------------------------------------------------------------

def parser():
    p = argparse.ArgumentParser(description="Run one step of the data pipeline (see data_pipeline/run_step.py)")
    p.add_argument("step", choices=STEPS)
    p.add_argument("name", nargs="?", help="participant only: the participant's zip, as the manifest names it")
    p.add_argument("--mode", choices=["sample", "full", "list"], default=None,
                   help="manifest only: who to process (data.mode in the config when left out)")
    p.add_argument("--config", default=None, help="Path to a config YAML")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.step == "participant" and not args.name:
        parser().error("participant needs the participant's zip name")
    cfg = load_config(args.config)
    try:
        if args.step == "manifest":
            manifest(cfg, args.mode)
        elif args.step == "participant":
            participant(cfg, args.name)
        else:
            {"patients": patients, "check_failures": check_failures, "build": build,
             "checks": run_checks, "notify_success": notify_success}[args.step](cfg)
    except STOPS as exc:
        logger.error("Step %s stopped: %s", args.step, exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
