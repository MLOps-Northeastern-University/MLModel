"""The contract every model follows, and the predictions it returns.

Owner: P2.

Every model (the baselines, the window model, any later one) subclasses Model, so the
train, tune, evaluate, bias and register steps only ever see this contract, never a
library. The base class owns the rules, so no model can skip them:

    - inputs are checked against the Dataset's schemas before fit and predict;
    - training and validation participants must not overlap;
    - predict and save refuse an unfitted model;
    - every prediction is checked: the model's labels, one row per hour of the data
      in its order, probabilities that add up to 1;
    - a window model (scores_windows) must also score every worn window of every
      participant, and an hourly model must not return windows;
    - save writes the weights and a manifest (what the model is, how it was trained,
      the code and library versions) and never overwrites; load refuses a manifest
      from another family or format.

A model family writes only the hooks at the bottom (_fit, _predict, _save_weights,
_load_weights), plus default_params if it has settings. Heavy
libraries (scikit-learn, PyTorch) are imported in the family's own file, never here.

    model = SomeFamily(targets=["sleep", "state", "tremor"], params={"C": 0.1}, spec=spec)
    model.fit(train, val=val, seed=0)
    predictions = model.predict(test)
    predictions.p_asleep("sleep_now")            # P(asleep) per hour
    predictions.p_state("state_now", "off")      # P(OFF) per hour, if awake
    model.save(folder)
    model = SomeFamily.load(folder)
"""

import datetime as dt
import json
import numbers
import shutil
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.core.data import GROUPS, Dataset, WindowSpec
from scripts.core.errors import ConfigError, ModelError, NotFittedError, SchemaError
from scripts.core.targets import (
    ASLEEP, HORIZONS, HOUR_INDEX, PARTICIPANT, STATE_GROUPS, get_target, label_name, parse_label)
from scripts.utils.file_io import read_json, write_json
from scripts.utils.logging_setup import environment, get_logger, log_step

logger = get_logger(__name__)

# Bump when the manifest's layout changes; load refuses other versions.
FORMAT_VERSION = 1
MANIFEST = "manifest.json"
WEIGHTS = "weights"
_SUM_TOLERANCE = 1e-6


# ---- predictions ------------------------------------------------------------------------------

@dataclass(frozen=True)
class WindowPredictions:
    """One participant's outputs per 10-s window: what the app and the timeline show.

    Attributes:
        time_s: (n,) each window's start, on the diary clock.
        probs: {label: (n, n_classes)} probabilities, labels as in the hourly predictions.
    """

    time_s: np.ndarray
    probs: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class Predictions:
    """A model's outputs: always per diary hour, optionally per window.

    Attributes:
        hours: participant_id and hour_index, one row per diary hour scored.
        probs: {label: (n_hours, n_classes)} class probabilities, such as
            {"sleep_now": (n, 2), "state_now": (n, 7), "tremor_now": (n, 3)}.
        windows: {participant: WindowPredictions}, for models that score windows.

    Raises:
        ModelError: An unknown label, a wrong shape, or probabilities that aren't
            non-negative and summing to 1. That's a bug in the model, not bad data.
    """

    hours: pd.DataFrame
    probs: Mapping[str, np.ndarray]
    windows: Optional[Mapping[str, WindowPredictions]] = None

    def __post_init__(self):
        missing = [c for c in (PARTICIPANT, HOUR_INDEX) if c not in self.hours.columns]
        if missing:
            raise ModelError(f"predictions need column(s) {missing} in hours")
        if not self.probs:
            raise ModelError("predictions have no labels")
        # Keep read-only copies, so nothing downstream can change a model's outputs by accident.
        hours = self.hours[[PARTICIPANT, HOUR_INDEX]].reset_index(drop=True)
        object.__setattr__(self, "hours", hours)
        object.__setattr__(self, "probs", {
            label: _checked_probs(label, p, len(hours), f"hourly {label}") for label, p in self.probs.items()})
        if self.windows is not None:
            object.__setattr__(self, "windows", {
                pid: self._checked_windows(pid, w) for pid, w in self.windows.items()})

    def _checked_windows(self, pid, w):
        n = len(w.time_s)
        unknown = set(w.probs) - set(self.probs)
        if unknown:
            raise ModelError(f"window predictions of {pid} have labels the hourly ones don't: {sorted(unknown)}")
        return WindowPredictions(_read_only(np.asarray(w.time_s)), {
            label: _checked_probs(label, p, n, f"window {label} of {pid}") for label, p in w.probs.items()})

    @property
    def labels(self):
        return tuple(self.probs)

    def predicted(self, label):
        """The most likely class index per hour."""
        return self._probs(label).argmax(axis=1)

    def p_asleep(self, label):
        """P(asleep) per hour for a sleep label, such as sleep_now."""
        target, _ = parse_label(label)
        if target != "sleep":
            raise ConfigError("p_asleep needs a sleep label, such as sleep_now")
        return self._probs(label)[:, ASLEEP]

    def p_state(self, label, group):
        """P(group) per hour for a state label: the sum of the group's classes, such as "off".
        The waking state, so it's P(group | awake); P(asleep) is the sleep output's."""
        target, _ = parse_label(label)
        if target != "state" or group not in STATE_GROUPS:
            raise ConfigError(f"p_state needs a state label and one of {sorted(STATE_GROUPS)}")
        return self._probs(label)[:, list(STATE_GROUPS[group])].sum(axis=1)

    def expected_score(self, label):
        """The expected diary score per hour: the sum of score x probability.

        Classes without a score are left out and the rest renormalised. For the state it's the
        expected kinesia score if awake; the sleep output has no score, so it's NaN there, as it
        is wherever every scored class has probability 0.
        """
        target, _ = parse_label(label)
        scores = get_target(target).scores
        scored = [i for i, s in enumerate(scores) if s is not None]
        p = self._probs(label)[:, scored]
        total = p.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(total > 0, p @ np.array([scores[i] for i in scored], float) / total, np.nan)

    def frame(self, label):
        """One label's probabilities as a table: participant_id, hour_index, then a column per class."""
        target, _ = parse_label(label)
        classes = pd.DataFrame(self._probs(label), columns=get_target(target).classes)
        return pd.concat([self.hours, classes], axis=1)

    def _probs(self, label):
        if label not in self.probs:
            raise ConfigError(f"no predictions for {label!r}; these have {sorted(self.probs)}")
        return self.probs[label]


def _checked_probs(label, p, n, where):
    try:
        target, _ = parse_label(label)
    except ConfigError as e:
        raise ModelError(f"{where}: {e}") from None
    k = len(get_target(target).classes)
    p = np.array(p, dtype=float)
    if p.shape != (n, k):
        raise ModelError(f"{where}: probabilities are {p.shape}, need ({n}, {k})")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ModelError(f"{where}: probabilities must be finite and non-negative")
    if n and not np.allclose(p.sum(axis=1), 1, atol=_SUM_TOLERANCE):
        raise ModelError(f"{where}: each row of probabilities must add up to 1")
    return _read_only(p)


def _read_only(a):
    a.flags.writeable = False
    return a


# ---- the model contract -----------------------------------------------------------------------

class Model(ABC):
    """The base class of every model family.

    Class attributes a family sets:
        family: Its name, as in the config and the registry, such as "logistic".
        needs: The Dataset groups it reads, from data.GROUPS, such as ("features",).
        scores_windows: True for a model that scores every 10-s window (its hourly
            predictions are those windows pooled per hour); False for one that only
            scores hours, such as the baselines on the hourly features.

    Args:
        targets: Target names from core/targets.py, such as ["state", "tremor"].
        horizons: Keys of targets.HORIZONS. Defaults to all of them.
        params: Settings, overriding default_params(). An unknown name is refused,
            so a typo in the config can't be silently ignored.
        spec: The WindowSpec from the config. Required if the family needs windows.

    Raises:
        ConfigError: An unknown target, horizon, group or setting, or a missing spec.
    """

    family: ClassVar[str] = ""
    needs: ClassVar[Tuple[str, ...]] = ()
    scores_windows: ClassVar[bool] = False

    def __init__(self, targets, horizons=tuple(HORIZONS), params=None, spec=None):
        if not self.family:
            raise ConfigError(f"{type(self).__name__} must set a family name")
        unknown = set(self.needs) - set(GROUPS)
        if unknown:
            raise ConfigError(f"{self.family} needs unknown group(s) {sorted(unknown)}; known: {GROUPS}")
        if self.scores_windows and "windows" not in self.needs:
            raise ConfigError(f"{self.family} scores windows, so it must need the windows group")
        self.targets, self.horizons = tuple(targets), tuple(horizons)
        if not self.targets or not self.horizons:
            raise ConfigError(f"{self.family}: name at least one target and one horizon")
        for name in self.targets:
            get_target(name)
        bad = [h for h in self.horizons if h not in HORIZONS]
        if bad:
            raise ConfigError(f"{self.family}: unknown horizon(s) {bad}; known: {sorted(HORIZONS)}")
        self.labels = tuple(label_name(t, h) for t in self.targets for h in self.horizons)
        self.params = self._merged_params(params)
        if "windows" in self.needs and spec is None:
            raise ConfigError(f"{self.family} reads windows, so it needs the window spec from the config")
        self.spec = spec
        self._training = None   # what fit recorded; None until fitted

    def _merged_params(self, params):
        defaults = self.default_params()
        params = dict(params or {})
        unknown = set(params) - set(defaults)
        if unknown:
            raise ConfigError(f"{self.family} has no setting(s) {sorted(unknown)}; known: {sorted(defaults)}")
        merged = {**defaults, **params}
        try:
            json.dumps(merged)   # settings go in the manifest, so they must be plain values
        except TypeError:
            raise ConfigError(f"{self.family}: settings must be numbers, text, lists or dicts") from None
        return merged

    def __repr__(self):
        state = "fitted" if self.is_fitted else "not fitted"
        return f"{type(self).__name__}(family={self.family!r}, labels={list(self.labels)}, {state})"

    @property
    def is_fitted(self):
        return self._training is not None

    @classmethod
    def default_params(cls):
        """The family's settings and their defaults. Override if it has any. What tuning searches
        among them is config, not code (config/model.yaml, tune:; steps/tune.py)."""
        return {}

    # ---- the public lifecycle ---------------------------------------------------------------

    def fit(self, train, val=None, sample_weight=None, seed=0):
        """Train on train's participants only, with val for early stopping or checks.

        Args:
            train: The training Dataset.
            val: Optional validation Dataset, with different participants.
            sample_weight: Optional weight per training hour (bias mitigation), in
                train.hours order: finite, 0 or more, not all 0.
            seed: Non-negative integer, for reproducible training.

        Returns:
            self, fitted.

        Raises:
            ConfigError: A bad seed or sample_weight.
            SchemaError: A Dataset is missing a group or label the model needs, or a
                participant is in both train and val.
        """
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
            raise ConfigError(f"seed must be a non-negative integer, not {seed!r}")
        self._check_data(train, "training", labels=True)
        if val is not None:
            self._check_data(val, "validation", labels=True)
            shared = set(train.participants) & set(val.participants)
            if shared:
                raise SchemaError(f"{len(shared)} participant(s) in both training and validation")
        weights = _checked_weights(sample_weight, len(train.hours))
        with log_step(logger, "fit", model=self.family):
            logger.info("fitting %s on %d participants, %d hours", self.family, len(train.participants),
                        len(train.hours), extra={
                            "event": "fit_data", "labels": list(self.labels), "seed": int(seed),
                            "train_participants": len(train.participants), "train_hours": len(train.hours),
                            "val_participants": len(val.participants) if val is not None else 0,
                            "weighted": weights is not None})
            self._fit(train, val, weights, int(seed))
        self._training = {
            "seed": int(seed), "train_participants": len(train.participants), "train_hours": len(train.hours),
            "val_participants": len(val.participants) if val is not None else 0,
            "weighted": weights is not None, "fitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        return self

    def predict(self, data):
        """Score every hour of data (and every window, if the family does).

        Returns:
            Predictions for exactly the model's labels, one row per hour of data, in its
            order; for a window model, also every worn window of every participant.

        Raises:
            NotFittedError: The model isn't fitted.
            SchemaError: data is missing a group the model needs.
            ModelError: The family returned predictions that break the contract.
        """
        self._require_fitted("predict")
        self._check_data(data, "scoring", labels=False)
        with log_step(logger, "predict", model=self.family):
            out = self._predict(data)
        if not isinstance(out, Predictions):
            raise ModelError(f"{self.family}._predict must return Predictions, not {type(out).__name__}")
        if set(out.labels) != set(self.labels):
            raise ModelError(f"{self.family} predicted {sorted(out.labels)}, but its labels are {sorted(self.labels)}")
        expected = data.hours[[PARTICIPANT, HOUR_INDEX]].reset_index(drop=True)
        if not out.hours.equals(expected):
            raise ModelError(f"{self.family} must predict one row per hour of the data, in the same order")
        self._check_windows(out, data)
        return out

    def _check_windows(self, out, data):
        """A window model scores every worn window, with all its labels; an hourly model none."""
        if not self.scores_windows:
            if out.windows is not None:
                raise ModelError(f"{self.family} doesn't score windows (scores_windows is False) but returned them")
            return
        if out.windows is None:
            raise ModelError(f"{self.family} scores windows but returned none")
        missing = set(data.participants) - set(out.windows)
        if missing:
            raise ModelError(f"{self.family} returned no windows for {len(missing)} participant(s)")
        for pid in data.participants:
            got, windows = out.windows[pid], data.windows[pid]
            if set(got.probs) != set(self.labels):
                raise ModelError(f"{self.family}: windows of {pid} need every label {sorted(self.labels)}")
            if not np.array_equal(got.time_s, windows.time_s[windows.worn]):
                raise ModelError(f"{self.family}: windows of {pid} must be every worn window, in order")

    def save(self, folder):
        """Write the weights and manifest to folder, which must not exist or be empty.

        Written to a temporary folder first and moved into place, so a crash never
        leaves a half-written model behind.

        Returns:
            The folder.
        """
        self._require_fitted("save")
        folder = Path(folder)
        if folder.exists() and any(folder.iterdir()):
            raise ModelError(f"{folder} isn't empty; models are never overwritten")
        folder.parent.mkdir(parents=True, exist_ok=True)
        tmp = folder.parent / f".{folder.name}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            (tmp / WEIGHTS).mkdir(parents=True)
            self._save_weights(tmp / WEIGHTS)
            write_json(self.manifest(), tmp / MANIFEST)
            if folder.exists():
                folder.rmdir()
            tmp.rename(folder)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        logger.info("saved %s to %s", self.family, folder, extra={"event": "model_saved", "path": str(folder)})
        return folder

    @classmethod
    def load(cls, folder):
        """Rebuild a saved model of this family.

        Raises:
            ModelError: No manifest, or one of another format or family.
        """
        folder = Path(folder)
        if not (folder / MANIFEST).exists():
            raise ModelError(f"no {MANIFEST} in {folder}")
        manifest = read_json(folder / MANIFEST)
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ModelError(f"{folder}: manifest format {manifest.get('format_version')}, "
                             f"this code reads {FORMAT_VERSION}")
        if manifest.get("family") != cls.family:
            raise ModelError(f"{folder} holds a {manifest.get('family')!r} model, not {cls.family!r}")
        spec = WindowSpec(**manifest["window_spec"]) if manifest.get("window_spec") else None
        model = cls(manifest["targets"], manifest["horizons"], manifest["params"], spec)
        model._load_weights(folder / WEIGHTS)
        model._training = manifest["training"]
        logger.info("loaded %s from %s", cls.family, folder, extra={"event": "model_loaded", "path": str(folder)})
        return model

    def manifest(self):
        """What the model is and how it was made, as saved next to its weights."""
        return {
            "format_version": FORMAT_VERSION,
            "family": self.family,
            "targets": list(self.targets),
            "horizons": list(self.horizons),
            "labels": list(self.labels),
            "needs": list(self.needs),
            "scores_windows": self.scores_windows,
            "params": self.params,
            "window_spec": asdict(self.spec) if self.spec else None,
            "training": self._training,
            "environment": environment(),
        }

    # ---- shared checks ------------------------------------------------------------------------

    def _check_data(self, data, purpose, labels):
        if not isinstance(data, Dataset):
            raise ConfigError(f"{purpose} data must be a Dataset, not {type(data).__name__}")
        data.check(self.needs, self.spec)
        missing = [label for label in self.labels if label not in data.labels.columns] if labels else []
        if missing:
            raise SchemaError(f"{purpose} data has no {missing} labels for {self.family}")

    def _require_fitted(self, action):
        if not self.is_fitted:
            raise NotFittedError(f"{self.family} must be fitted before {action}")

    # ---- what a model family writes ------------------------------------------------------------

    @abstractmethod
    def _fit(self, train, val, sample_weight, seed):
        """Learn from train (checked), with val (checked, or None) and weights (array or None)."""

    @abstractmethod
    def _predict(self, data):
        """Return Predictions for self.labels, one row per hour of data in its order."""

    @abstractmethod
    def _save_weights(self, folder):
        """Write whatever the family needs to rebuild itself into folder (it exists and is empty)."""

    @abstractmethod
    def _load_weights(self, folder):
        """Read what _save_weights wrote."""


def _checked_weights(sample_weight, n):
    if sample_weight is None:
        return None
    w = np.asarray(sample_weight, dtype=float)
    if w.shape != (n,) or not np.isfinite(w).all() or (w < 0).any() or not w.sum() > 0:
        raise ConfigError(f"sample_weight must be {n} finite weights of 0 or more, not all 0")
    return w
