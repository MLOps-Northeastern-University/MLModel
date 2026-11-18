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
    export     chosen.json, the data                      bundle/
    register   bundle/                                    the registry's next version

Every step takes (cfg, tracker, data, split). data and split are None in a real run, so each step
reads them from the config's paths; tests pass small ones.
"""

from pathlib import Path

from scripts.core.errors import SchemaError
from scripts.data.folds import read_split
from scripts.data.loaders import load_dataset
from scripts.steps.bias import CheckBias
from scripts.steps.compare_production import CompareProduction
from scripts.steps.evaluate import EvaluateModel
from scripts.steps.explain import ExplainModels
from scripts.steps.export import ExportModel
from scripts.steps.register import RegisterModel, active_metrics
from scripts.steps.select import SelectModel
from scripts.steps.train import TrainModels
from scripts.steps.tune import TuneModels
from scripts.utils.file_io import read_json, write_json

CANDIDATES_FILE = "candidates.json"
CHOSEN_FILE = "chosen.json"


# ---- the hand-off files -------------------------------------------------------------------

def artifacts(cfg):
    return Path(cfg["resolved_paths"]["artifacts_dir"])


def read_candidates(cfg):
    """The candidates as the last step left them."""
    path = artifacts(cfg) / CANDIDATES_FILE
    if not path.exists():
        raise SchemaError(f"no {path}: run train first")
    return read_json(path)


def write_candidates(cfg, candidates):
    write_json(candidates, artifacts(cfg) / CANDIDATES_FILE)


def read_chosen(cfg):
    """The candidate select chose."""
    path = artifacts(cfg) / CHOSEN_FILE
    if not path.exists():
        raise SchemaError(f"no {path}: run select first")
    return read_json(path)


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


# ---- the steps ----------------------------------------------------------------------------

def tune(cfg, tracker, data=None, split=None):
    TuneModels(cfg, tracker).run(data, split)


def train(cfg, tracker, data=None, split=None):
    (artifacts(cfg) / CHOSEN_FILE).unlink(missing_ok=True)       # never left over from an earlier run
    TrainModels(cfg, tracker).run(data, split)


def evaluate(cfg, tracker, data=None, split=None):
    # Saved, so bias (in its own container) sees which candidates passed the validation gate.
    write_candidates(cfg, EvaluateModel(cfg, tracker).run(read_candidates(cfg)))


def bias(cfg, tracker, data=None, split=None):
    CheckBias(cfg, tracker).run(read_candidates(cfg), data, split)    # saves candidates.json itself


def explain(cfg, tracker, data=None, split=None):
    ExplainModels(cfg, tracker).run(read_candidates(cfg), data, split)


def select(cfg, tracker, data=None, split=None):
    chosen = SelectModel(cfg, tracker).run(read_candidates(cfg))
    write_json(chosen, artifacts(cfg) / CHOSEN_FILE)


def compare(cfg, tracker, data=None, split=None):
    family_dir = RegisterModel(cfg, tracker).family_dir
    CompareProduction(cfg, tracker).run(read_chosen(cfg)["metrics"], active_metrics(family_dir))


def export(cfg, tracker, data=None, split=None):
    stats = training_stats(cfg, split or read_split(cfg), data)
    ExportModel(cfg, tracker).run(read_chosen(cfg), stats)


def register(cfg, tracker, data=None, split=None):
    """Release the bundle as the next version, made active. Returns the version."""
    bundle_dir = ExportModel(cfg, tracker).bundle_dir
    if not (bundle_dir / "manifest.json").exists():
        raise SchemaError(f"no bundle in {bundle_dir}: run export first")
    return RegisterModel(cfg, tracker).run(bundle_dir)


# The steps in order, with their owners (for the logs). tune runs only when asked.
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
