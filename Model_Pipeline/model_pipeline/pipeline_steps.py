"""The model pipeline's steps, each runnable on its own, handing over through files in artifacts/.

The whole run (run_model_pipeline.py) calls them in order in one process. The Airflow DAG runs each
in its own container (run_step.py). Both go through these functions, so a step does exactly the same
work either way, and no result passes between steps in memory:

    step       reads                                      writes
    tune       the data                                   tuned_params.json
    train      the data, tuned_params.json                candidates.json, predictions/, models/
    evaluate   candidates.json, predictions/              candidates.json (+ metrics, the validation gate)
    bias       candidates.json, predictions/, the data    candidates.json (+ the bias gate), bias/
    explain    candidates.json, the data                  explain/
    select     candidates.json                            chosen.json
    compare    chosen.json, the active version's metrics  (nothing: it passes or stops the run)
    export     chosen.json, the data, predictions/        bundle/ (its stats.json: the drift references)
    register   bundle/                                    the registry's next version

Every step takes (cfg, tracker, data, split). data and split are None in a real run, so each step
reads them from the config's paths; tests pass small ones.
"""

from pathlib import Path

import numpy as np

from model_pipeline.core.errors import SchemaError
from model_pipeline.core.targets import STATE_GROUPS, get_target
from model_pipeline.data.folds import read_split
from model_pipeline.data.loaders import load_dataset
from model_pipeline.steps.bias import CheckBias, read_predictions
from model_pipeline.steps.compare_production import CompareProduction
from model_pipeline.steps.evaluate import EvaluateModel
from model_pipeline.steps.explain import ExplainModels
from model_pipeline.steps.export import ExportModel
from model_pipeline.steps.register import RegisterModel, active_metrics
from model_pipeline.steps.select import SelectModel
from model_pipeline.steps.train import TrainModels
from model_pipeline.steps.tune import TuneModels
from model_pipeline.utils.file_io import read_json, write_json

CANDIDATES_FILE = "candidates.json"
CHOSEN_FILE = "chosen.json"


# ---- the hand-off files -------------------------------------------------------------------

def artifacts(cfg):
    """The folder every hand-off file is in (paths.artifacts_dir), which load_config creates."""
    return Path(cfg["resolved_paths"]["artifacts_dir"])


def read_candidates(cfg):
    """The candidates as the last step left them."""
    path = artifacts(cfg) / CANDIDATES_FILE
    # A missing file is a SchemaError, a PipelineError, so a step run out of order (in Airflow or by hand)
    # stops with exit code 1 and this message instead of crashing on a FileNotFoundError.
    if not path.exists():
        raise SchemaError(f"no {path}: run train first")
    return read_json(path)


def write_candidates(cfg, candidates):
    """Save the candidates for the steps after this one (evaluate does; bias saves them itself)."""
    write_json(candidates, artifacts(cfg) / CANDIDATES_FILE)


def read_chosen(cfg):
    """The candidate select chose."""
    path = artifacts(cfg) / CHOSEN_FILE
    if not path.exists():
        raise SchemaError(f"no {path}: run select first")
    return read_json(path)


def training_stats(cfg, split, data=None):
    """The 7 hourly features' statistics over the development patients, for drift checks."""
    # Development patients only: the test patients are never used for any statistic (data/folds.py), and the
    # final model was fitted on every development patient. A real run loads just the features group.
    if data is None:
        data = load_dataset(cfg, groups=("features",), participants=split.development)
    else:
        data = data.subset(split.development)
    table = data.features.astype(float)
    return {"rows": len(table), "participants": len(split.development),
            "features": {name: {"mean": col.mean(), "std": col.std(), "min": col.min(),
                                "p50": col.median(), "max": col.max()}
                         for name, col in table.items()}}


# How finely the reference splits P(OFF): ten bands, 0-0.1 up to 0.9-1.
P_OFF_BANDS = 10


def prediction_reference(cfg, name):
    """How often the candidate said each P(OFF) band on its out-of-fold development hours with a worn
    window: what the Backend's drift check compares the live hours with (prediction drift, PSI).

    Counts of hours only, no patient IDs and nothing per patient. None if train wrote no state_now
    predictions for this candidate.
    """
    table = read_predictions(artifacts(cfg) / "predictions", ["state_now"]).get("state_now")
    if table is None:
        return None
    # This candidate's out-of-fold rows only: predictions/ holds every candidate, and the test fold (when
    # train scored it) is left out. An hour without a worn window gets the training class mix rather than a
    # real prediction (experiments.py), so it would only blur the reference. Unlabelled hours stay in: the
    # live hours the Backend compares with have no diary answers either.
    rows = table[(table["model"] == name) & (table["fold"] != "test") & table["worn"]]
    if rows.empty:
        return None
    # P(OFF) is the sum of the three akinesia classes, kinesia scores -3 to -1 (core/targets.STATE_GROUPS).
    classes = list(get_target("state").classes)
    p_off = rows[[classes[i] for i in STATE_GROUPS["off"]]].sum(axis=1).to_numpy()
    # np.histogram's last band is closed, so a P(OFF) of exactly 1.0 is counted in 0.9-1 rather than lost.
    counts, edges = np.histogram(p_off, bins=P_OFF_BANDS, range=(0.0, 1.0))
    return {"output": "p_off", "label": "state_now", "hours": int(len(p_off)),
            "bands": [round(float(e), 2) for e in edges], "counts": [int(c) for c in counts]}


# ---- the steps ----------------------------------------------------------------------------

def tune(cfg, tracker, data=None, split=None):
    TuneModels(cfg, tracker).run(data, split)


def train(cfg, tracker, data=None, split=None):
    # TrainModels clears candidates.json and predictions/ itself, but chosen.json belongs to select. Removing
    # it here means that if this run stops before select, compare and export find no choice and stop, rather
    # than releasing the model an earlier run chose.
    (artifacts(cfg) / CHOSEN_FILE).unlink(missing_ok=True)       # never left over from an earlier run
    TrainModels(cfg, tracker).run(data, split)


def evaluate(cfg, tracker, data=None, split=None):
    # Saved, so bias (in its own container) sees which candidates passed the validation gate.
    write_candidates(cfg, EvaluateModel(cfg, tracker).run(read_candidates(cfg)))


def bias(cfg, tracker, data=None, split=None):
    # Because bias comes before explain, a run in which no candidate passes the bias gate stops here and
    # never spends time on explain, which is a report and never a gate.
    CheckBias(cfg, tracker).run(read_candidates(cfg), data, split)    # saves candidates.json itself


def explain(cfg, tracker, data=None, split=None):
    # Writes its reports under artifacts/explain/ only, leaving candidates.json as bias saved it for select.
    ExplainModels(cfg, tracker).run(read_candidates(cfg), data, split)


def select(cfg, tracker, data=None, split=None):
    chosen = SelectModel(cfg, tracker).run(read_candidates(cfg))
    # chosen.json is the whole candidate entry (name, model folder, metrics, run id): everything compare and
    # export need, so neither reads candidates.json.
    write_json(chosen, artifacts(cfg) / CHOSEN_FILE)


def compare(cfg, tracker, data=None, split=None):
    # RegisterModel is built only for family_dir, which uses MODEL_REGISTRY when it is set (gs:// in CI), so
    # the gate compares with the version active in the registry register writes to. Before the first release
    # active_metrics is None, and CompareProduction lets the model through as the first one.
    family_dir = RegisterModel(cfg, tracker).family_dir
    CompareProduction(cfg, tracker).run(read_chosen(cfg)["metrics"], active_metrics(family_dir))


def export(cfg, tracker, data=None, split=None):
    chosen = read_chosen(cfg)
    # Run on its own (run_step) the step is given no split, so it is read from the config's split file.
    stats = training_stats(cfg, split or read_split(cfg), data)
    # The Backend's drift check compares live P(OFF) with this (stats.json in the bundle).
    stats["prediction_reference"] = prediction_reference(cfg, chosen["name"])
    ExportModel(cfg, tracker).run(chosen, stats)


def register(cfg, tracker, data=None, split=None):
    """Release the bundle as the next version, made active. Returns the version."""
    bundle_dir = ExportModel(cfg, tracker).bundle_dir
    # This only checks that a bundle exists. It is this run's because the steps run in order (in one process,
    # or with Airflow skipping every task after a failed one) and export rebuilds bundle/ from scratch.
    if not (bundle_dir / "manifest.json").exists():
        raise SchemaError(f"no bundle in {bundle_dir}: run export first")
    return RegisterModel(cfg, tracker).run(bundle_dir)


# The steps in order, with their owners (for the logs). tune runs only when asked.
# The keys are also run_step's command names (its parser offers them as choices), and the owner goes on
# every log line of the step (log_step), so a failure shows whose code to look at.
STEPS = {
    "tune": (tune, "P2"),
    "train": (train, "P2"),
    "evaluate": (evaluate, "P1"),
    "bias": (bias, "P3"),
    "explain": (explain, "P3"),
    "select": (select, "P4"),
    "compare": (compare, "P4"),
    "export": (export, "P4"),
    "register": (register, "P4"),
}


def steps_for(tune=False, release=True):
    """The steps a run goes through, in order."""
    names = list(STEPS)
    if not tune:
        names.remove("tune")
    if not release:
        names.remove("register")
    return names
