"""Reads the data pipeline's outputs into a Dataset (core/data.py).

Owner: P1 (component A).

The data pipeline writes paths.processed_data (Data_Pipeline/data/processed/):

    hours.csv          one row per diary hour: the diary, bias slices, W_* features
    events.csv         one row per dose
    patients.csv       one row per participant: wrist, more_affected_side, dbs
    windows/<pid>/     time_s.npy, worn.npy, signal.npy (n, axes, samples) float32
    dataset_meta.json  the window spec it was cut with, and counts

load_dataset reads them (the windows memory-mapped, so nothing is copied),
derives the labels with core/targets.py, and checks every group. data_version
names the data, from the data pipeline's dvc.lock, for the tracker.
"""

import hashlib

import numpy as np
import yaml

from scripts.core.data import GROUPS, Dataset, Windows, WindowSpec
from scripts.core.targets import TARGETS, build_labels
from scripts.utils.file_io import read_json, read_table
from scripts.utils.logging_setup import get_logger

logger = get_logger(__name__)

WINDOW_ARRAYS = ("time_s", "worn", "signal")


def load_dataset(cfg, check=True):
    """The data pipeline's output as one checked Dataset, every group loaded.

    Raises:
        FileNotFoundError: The data isn't there; run `dvc pull` or the data pipeline.
        SchemaError: A group doesn't match its schema, or the windows were cut
            with another spec than cfg["windows"].
    """
    folder = cfg["resolved_paths"]["processed_data"]
    if not (folder / "dataset_meta.json").exists():
        raise FileNotFoundError(f"no dataset in {folder}: run `dvc pull` or the data pipeline in ../Data_Pipeline")
    meta = read_json(folder / "dataset_meta.json")
    spec = WindowSpec(meta["spec_version"], meta["seconds"], meta["rate_hz"], meta["axes"])

    table = read_table(folder / "hours.csv")
    feature_columns = [cfg["data"]["feature_prefix"] + name for name in cfg["data"]["features"]]
    patients = read_table(folder / "patients.csv")
    for column in ("more_affected_side", "dbs"):
        patients[column] = _bool_column(patients[column])
    dataset = Dataset(
        spec=spec,
        hours=table.drop(columns=feature_columns),
        labels=build_labels(table, cfg["data"].get("targets") or list(TARGETS)),
        features=table[feature_columns],
        windows={pid: _windows(folder / "windows" / pid) for pid in table["participant_id"].unique()},
        events=read_table(folder / "events.csv"),
        patients=patients,
    )
    if check:
        dataset.check(GROUPS, WindowSpec.from_config(cfg))
    logger.info("loaded %d participants, %d hours, %d windows from %s", len(dataset.participants),
                len(dataset.hours), sum(len(w) for w in dataset.windows.values()), folder)
    return dataset


def _windows(folder):
    if not folder.is_dir():
        raise FileNotFoundError(f"no windows for {folder.name} in {folder.parent}")
    return Windows(*(np.load(folder / f"{name}.npy", mmap_mode="r") for name in WINDOW_ARRAYS))


def data_version(cfg):
    """A short id for the data, from the hashes of its outputs in the data pipeline's dvc.lock.

    The same data gives the same id on every machine. None when there's no dvc.lock.
    It names the data the lock describes, so run `dvc status` if files were changed by hand.
    """
    path = cfg["resolved_paths"]["dvc_lock"]
    if not path.exists():
        return None
    stages = yaml.safe_load(path.read_text())["stages"]
    outs = sorted(f"{o['path']}:{o['md5']}" for stage in stages.values() for o in stage.get("outs", []))
    return hashlib.sha256("\n".join(outs).encode()).hexdigest()[:12]


def _bool_column(values):
    """A CSV's True/False column as booleans. A blank stays NaN, so the patients check refuses it."""
    return values.map({True: True, False: False, "True": True, "False": False}).astype(object)
