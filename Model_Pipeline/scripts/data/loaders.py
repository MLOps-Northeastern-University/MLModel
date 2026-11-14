"""Reads the data pipeline's outputs into a Dataset (core/data.py).

Owner: P1 (component A).

    data = load_dataset(cfg, groups=("windows", "events", "patients"))

Only the groups a model asks for are read, so a baseline on the hourly features never
touches the windows. hours and labels are always read: every diary hour, with each
target's label derived as core/targets.py says. Windows are memory-mapped one participant
at a time, on first use, so a Dataset of every patient takes almost no memory.

The files, from config paths (the layout the data pipeline writes):

    hours_csv        every diary hour: answers, kinesia_text, scores, bias columns
    features_csv     the hourly features (features group)
    windows_folder   <participant>/signal.npy, time_s.npy, worn.npy, and manifest.json
    events_csv       one row per dose or check-in
    patients_csv     one row per participant: more_affected_side, dbs
"""

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.core.data import GROUPS, Dataset, Windows, WindowSpec
from scripts.core.errors import ConfigError, SchemaError
from scripts.core.targets import HOUR_INDEX, PARTICIPANT, TARGETS, build_labels
from scripts.utils.logging_setup import get_logger

logger = get_logger(__name__)

MANIFEST = "manifest.json"
WINDOW_FILES = ("signal.npy", "time_s.npy", "worn.npy")


class WindowStore(Mapping):
    """{participant: Windows}, read from the windows folder when first asked for, memory-mapped."""

    def __init__(self, folder, participants):
        self.folder = Path(folder)
        self._participants = tuple(p for p in participants if (self.folder / p / "worn.npy").exists())
        self._cache = {}

    def __getitem__(self, participant):
        if participant not in self._participants:
            raise KeyError(participant)
        if participant not in self._cache:
            arrays = [np.load(self.folder / participant / f, mmap_mode="r") for f in WINDOW_FILES]
            signal, time_s, worn = arrays
            self._cache[participant] = Windows(time_s=time_s, worn=worn, signal=signal)
        return self._cache[participant]

    def __iter__(self):
        return iter(self._participants)

    def __len__(self):
        return len(self._participants)


def read_window_spec(folder):
    """The WindowSpec the windows were cut with, from the folder's manifest.

    Raises:
        SchemaError: No manifest, or one without the spec.
    """
    path = Path(folder) / MANIFEST
    if not path.exists():
        raise SchemaError(f"no {MANIFEST} in {folder}: can't tell how the windows were cut")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    try:
        values = {k: manifest[k] for k in ("spec_version", "seconds", "rate_hz", "axes")}
        try:
            return WindowSpec.from_config({"windows": values})     # the same checks as the config's spec
        except ConfigError as e:
            raise SchemaError(f"{path}: {e}") from None
    except KeyError as missing:
        raise SchemaError(f"{path} has no {missing}") from None


def load_dataset(cfg, groups=(), participants=None, targets=tuple(TARGETS)):
    """Read the hours, labels and the input groups asked for into a Dataset.

    Args:
        cfg: The pipeline config (paths from cfg["resolved_paths"]).
        groups: Input groups to read, from data.GROUPS. hours and labels are always read.
        participants: Only these participants, or everyone in the hours table.
        targets: The targets to derive labels for.

    Returns:
        A Dataset. Its spec is the windows' own (from their manifest) when windows are
        read, so Dataset.check can refuse windows cut another way; else the config's.

    Raises:
        ConfigError: An unknown group, or a path missing from the config.
        SchemaError: A file that's missing or doesn't match its schema.
    """
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise ConfigError(f"unknown input group(s) {sorted(unknown)}; known: {GROUPS}")
    paths = cfg["resolved_paths"]

    hours = _read_table(paths, "hours_csv")
    if participants is not None:
        wanted = set(participants)
        missing = wanted - set(hours[PARTICIPANT])
        if missing:
            raise SchemaError(f"{len(missing)} participant(s) aren't in the hours table, e.g. {min(missing)}")
        hours = hours[hours[PARTICIPANT].isin(wanted)]
    hours = hours.sort_values([PARTICIPANT, HOUR_INDEX]).reset_index(drop=True)
    labels = build_labels(hours, list(targets))
    people = sorted(hours[PARTICIPANT].unique())

    spec = WindowSpec.from_config(cfg)
    parts = {}
    if "features" in groups:
        names = cfg["data"]["features"]
        prefix = cfg["data"].get("feature_prefix", "")
        table = _read_table(paths, "features_csv")
        columns = [prefix + n for n in names]
        absent = [c for c in columns if c not in table.columns]
        if absent:
            raise SchemaError(f"features_csv has no {absent}")
        if table.duplicated([PARTICIPANT, HOUR_INDEX]).any():
            raise SchemaError(f"features_csv has a participant's {HOUR_INDEX} twice")
        merged = hours[[PARTICIPANT, HOUR_INDEX]].merge(
            table[[PARTICIPANT, HOUR_INDEX, *columns]], on=[PARTICIPANT, HOUR_INDEX], how="left")
        parts["features"] = merged[columns].set_axis(names, axis=1).set_axis(hours.index)
    if "windows" in groups:
        folder = _path(paths, "windows_folder")
        spec = read_window_spec(folder)
        parts["windows"] = WindowStore(folder, people)
    if "events" in groups:
        events = _read_table(paths, "events_csv")
        parts["events"] = events[events[PARTICIPANT].isin(people)].reset_index(drop=True)
    if "patients" in groups:
        patients = _read_table(paths, "patients_csv")
        parts["patients"] = patients[patients[PARTICIPANT].isin(people)].reset_index(drop=True)

    data = Dataset(spec=spec, hours=hours, labels=labels, **parts)
    logger.info("loaded %d participants, %d hours", len(people), len(hours), extra={
        "event": "dataset_loaded", "participants": len(people), "hours": len(hours),
        "groups": sorted(parts), "window_spec": spec.version})
    return data


def _path(paths, key):
    if key not in paths:
        raise ConfigError(f"config paths need {key}")
    path = Path(paths[key])
    if not path.exists():
        raise SchemaError(f"{key} not found: {path} (run `dvc pull` in Data_Pipeline?)")
    return path


def _read_table(paths, key):
    return pd.read_csv(_path(paths, key), low_memory=False)
