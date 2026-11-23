"""What a model is trained and scored on: the Dataset and the shape of each input.

Owner: P2, with P1 (who fills it in data/loaders.py).

This module is the contract only: it defines the input groups, holds them in a
Dataset, and checks them. Reading files into a Dataset is data/loaders.py.

    hours      one row per diary hour: participant_id, hour_index, bias slices
    labels     targets.build_labels(hours): state_now, tremor_now, ...
    features   the hourly features, one row per hour     (the baselines)
    windows    {participant: Windows}, the 10-s windows   (the window model)
    events     every dose and check-in, one row each
    patients   one row per participant: side, DBS

A model asks for the groups it needs, and check() fails clearly if one is
missing or has the wrong shape, before any training starts.

One clock for everything: seconds since the participant's first diary day
began. hour_index is on the same clock, so a window or event at time_s belongs
to hour time_s // 3600.
"""

from dataclasses import dataclass, replace
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from model_pipeline.core.errors import ConfigError, SchemaError
from model_pipeline.core.targets import HOUR_INDEX, KINESIA_SCORES, LABEL_NAMES, PARTICIPANT, TREMOR_SCORES
from model_pipeline.utils.logging_setup import get_logger

logger = get_logger(__name__)

GROUPS = ("features", "windows", "events", "patients")
SECONDS_PER_HOUR = 3600

# Events: one row per dose or check-in.
EVENT_COLUMNS = ("participant_id", "time_s", "kind", "ledd", "source", "kinesia_score", "tremor_score")
EVENT_KINDS = ("dose", "check_in")
# How much to trust a dose's time. none_logged is a scheduled dose with no log;
# its ledd is the scheduled amount.
DOSE_SOURCES = ("on_time", "late", "none_logged")

PATIENT_COLUMNS = ("participant_id", "more_affected_side", "dbs")


@dataclass(frozen=True)
class WindowSpec:
    """How the signal is cut into windows. Read from config, never hard-coded.

    The data pipeline and the app must cut windows the same way, so the spec has
    a version that every Dataset records and every model checks.
    """

    version: int
    seconds: int
    rate_hz: int
    axes: int

    @property
    def samples(self):
        """Samples per window per axis."""
        return self.seconds * self.rate_hz

    @classmethod
    def from_config(cls, cfg):
        """Build the spec from cfg["windows"].

        Raises:
            ConfigError: The section or a value is missing, or a value isn't a positive whole number.
        """
        section = cfg.get("windows")
        if not isinstance(section, dict):
            raise ConfigError("config needs a windows section: spec_version, seconds, rate_hz, axes")
        values = {}
        for key in ("spec_version", "seconds", "rate_hz", "axes"):
            value = section.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigError(f"windows.{key} must be a positive whole number, not {value!r}")
            values[key] = value
        return cls(values["spec_version"], values["seconds"], values["rate_hz"], values["axes"])

    def __str__(self):
        return f"v{self.version} ({self.seconds} s at {self.rate_hz} Hz, {self.axes} axes)"


@dataclass(frozen=True)
class Windows:
    """One participant's movement, cut into windows.

    The arrays may be memory-mapped, so a Dataset of every patient fits in memory.

    Attributes:
        time_s: (n,) start of each window, in seconds on the diary clock, increasing.
        worn: (n,) True where the watch was worn. Unworn windows are never scored.
        signal: (n, axes, samples) float32, in g, at the spec's rate, each axis's mean removed.
    """

    time_s: np.ndarray
    worn: np.ndarray
    signal: np.ndarray

    def __len__(self):
        return len(self.time_s)

    def hour_index(self):
        """The diary hour each window belongs to."""
        return (self.time_s // SECONDS_PER_HOUR).astype(int)

    def check(self, spec, participant):
        """Raise SchemaError if the arrays don't match each other or the spec."""
        where = f"windows of {participant}"
        n = len(self.time_s)
        expected = (n, spec.axes, spec.samples)
        if self.signal.shape != expected:
            raise SchemaError(f"{where}: signal is {self.signal.shape}, spec {spec} needs {expected}")
        if self.signal.dtype != np.float32:
            raise SchemaError(f"{where}: signal must be float32, not {self.signal.dtype}")
        if self.worn.shape != (n,) or self.worn.dtype != bool:
            raise SchemaError(f"{where}: worn must be {n} booleans, not {self.worn.shape} {self.worn.dtype}")
        # Windows can't overlap: each starts at least one window length after the last.
        # Values inside signal are the data pipeline's to check; scanning them here would read every file.
        if n and not (np.isfinite(self.time_s).all() and (np.diff(self.time_s) >= spec.seconds).all()):
            raise SchemaError(f"{where}: time_s must be finite and increase by at least {spec.seconds} s")


@dataclass(frozen=True)
class Dataset:
    """Everything a model is trained or scored on, for a set of participants.

    hours and labels are always there; the input groups are None unless loaded.
    Build one with data/loaders.py, then call check() with the groups the model needs.
    """

    spec: WindowSpec
    hours: pd.DataFrame
    labels: pd.DataFrame
    features: Optional[pd.DataFrame] = None
    windows: Optional[Mapping[str, Windows]] = None
    events: Optional[pd.DataFrame] = None
    patients: Optional[pd.DataFrame] = None

    def __post_init__(self):
        _require_columns(self.hours, (PARTICIPANT, HOUR_INDEX), "hours")
        hour = pd.to_numeric(self.hours[HOUR_INDEX], errors="coerce")
        if hour.isna().any() or not (hour == hour.round()).all():
            raise SchemaError(f"hours: every {HOUR_INDEX} must be a whole number")
        if self.hours.duplicated([PARTICIPANT, HOUR_INDEX]).any():
            raise SchemaError(f"hours: a participant has the same {HOUR_INDEX} twice")
        if not self.labels.index.equals(self.hours.index):
            raise SchemaError("labels must have one row per hour, on the same index as hours")
        unknown = set(self.labels.columns) - LABEL_NAMES
        if unknown:
            raise SchemaError(f"labels: unknown label(s) {sorted(unknown)}; known: {sorted(LABEL_NAMES)}")

    @property
    def participants(self):
        """The participants in this Dataset, sorted."""
        return tuple(sorted(self.hours[PARTICIPANT].unique()))

    def subset(self, participants):
        """The same Dataset for only these participants, such as one fold's training set.

        Raises:
            SchemaError: A participant isn't in this Dataset.
        """
        wanted = set(participants)
        missing = wanted - set(self.participants)
        if missing:
            raise SchemaError(f"subset: {len(missing)} participant(s) not in this dataset, e.g. {min(missing)}")
        keep = self.hours[PARTICIPANT].isin(wanted)

        def rows(table):
            return None if table is None else table[table[PARTICIPANT].isin(wanted)]

        return replace(
            self,
            hours=self.hours[keep],
            labels=self.labels[keep],
            features=None if self.features is None else self.features[keep],
            # Only the wanted participants' windows are opened: windows can be a lazy store of every patient.
            windows=None if self.windows is None else {p: self.windows[p] for p in sorted(wanted)
                                                       if p in set(self.windows.keys())},
            events=rows(self.events),
            patients=rows(self.patients),
        )

    def check(self, needs, spec):
        """Check that the groups a model needs are here and match the model's spec.

        Args:
            needs: Group names from GROUPS, such as ("windows", "events", "patients").
            spec: The WindowSpec the model was built for, from its config.

        Raises:
            ConfigError: An unknown group name.
            SchemaError: A group is missing or doesn't match its schema, or the
                windows were cut with a different spec.
        """
        unknown = set(needs) - set(GROUPS)
        if unknown:
            raise ConfigError(f"unknown input group(s) {sorted(unknown)}; known: {GROUPS}")
        for name in needs:
            if getattr(self, name) is None:
                raise SchemaError(f"the model needs the {name} group, but it isn't loaded")
        if "windows" in needs and self.spec != spec:
            raise SchemaError(f"windows were cut with spec {self.spec}, but the model expects {spec}")

        if "features" in needs:
            self._check_features()
        if "windows" in needs:
            self._check_windows()
        if "events" in needs:
            self._check_events()
        if "patients" in needs:
            self._check_patients()
        logger.info("dataset checked: %d participants, %d hours", len(self.participants), len(self.hours),
                    extra={"event": "dataset_checked", "participants": len(self.participants),
                           "hours": len(self.hours), "groups": list(needs), "window_spec": self.spec.version})

    def _check_features(self):
        if not self.features.index.equals(self.hours.index):
            raise SchemaError("features must have one row per hour, on the same index as hours")
        not_numeric = [c for c in self.features.columns if not pd.api.types.is_numeric_dtype(self.features[c])]
        if not_numeric:
            raise SchemaError(f"features must be numbers; not numeric: {not_numeric}")

    def _check_windows(self):
        missing = set(self.participants) - set(self.windows)
        if missing:
            raise SchemaError(f"windows: {len(missing)} participant(s) have none, e.g. {min(missing)}")
        for participant in self.participants:
            self.windows[participant].check(self.spec, participant)

    def _check_events(self):
        events = self.events
        _require_columns(events, EVENT_COLUMNS, "events")
        _require_known_participants(events, self.participants, "events")
        # Non-numbers become NaN here, so they fail the same checks as missing values.
        if not np.isfinite(pd.to_numeric(events["time_s"], errors="coerce")).all():
            raise SchemaError("events: every event needs a finite time_s")
        _require_values(events["kind"], EVENT_KINDS, "events.kind")

        doses = events[events["kind"] == "dose"]
        ledd = pd.to_numeric(doses["ledd"], errors="coerce")
        if not (np.isfinite(ledd) & (ledd >= 0)).all():
            raise SchemaError("events: every dose needs an LEDD of 0 or more")
        _require_values(doses["source"], DOSE_SOURCES, "events.source")

        check_ins = events[events["kind"] == "check_in"]
        if (check_ins["kinesia_score"].isna() & check_ins["tremor_score"].isna()).any():
            raise SchemaError("events: a check-in has neither answer")
        _require_values(check_ins["kinesia_score"].dropna(), KINESIA_SCORES, "events.kinesia_score")
        _require_values(check_ins["tremor_score"].dropna(), TREMOR_SCORES, "events.tremor_score")

    def _check_patients(self):
        patients = self.patients
        _require_columns(patients, PATIENT_COLUMNS, "patients")
        if patients[PARTICIPANT].duplicated().any():
            raise SchemaError("patients: a participant has more than one row")
        missing = set(self.participants) - set(patients[PARTICIPANT])
        if missing:
            raise SchemaError(f"patients: {len(missing)} participant(s) have no row, e.g. {min(missing)}")
        for column in ("more_affected_side", "dbs"):
            _require_values(patients[column], (True, False), f"patients.{column}")


def _require_columns(table, columns, name):
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise SchemaError(f"{name} needs column(s) {missing}")


def _require_known_participants(table, participants, name):
    unknown = set(table[PARTICIPANT]) - set(participants)
    if unknown:
        raise SchemaError(f"{name}: {len(unknown)} participant(s) not in hours, e.g. {min(unknown)}")


def _require_values(values, allowed, name):
    bad = ~values.isin(allowed)
    if bad.any():
        raise SchemaError(f"{name} must be one of {allowed}, not {values[bad].iloc[0]!r}")
