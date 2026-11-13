"""Window classifiers: logistic regression and gradient boosting on each 10-s window, with no past.

Owner: P2.

The baselines the motor causal transformer has to beat (MODEL.md, Experiment order, stage 0). They
follow the prototype's best version (team-hub/exploration/baselines/window_classifier.py, harnet10
features only):

    - every worn window of an answered hour takes that hour's label;
    - one classifier per label (a target at a horizon) reads the window's 1,024 harnet10 features
      and nothing else: no doses, no patient fields, no past;
    - an hour's prediction is the mean of its worn windows' probabilities, and an hour with no worn
      windows gets the training hours' class mix.

Two families, which differ only in the classifier:

    window_logistic   standardise, then multinomial logistic regression
    window_boosting   standardise, PCA to 64 components, then histogram gradient boosting

    model = WindowLogistic(targets=["sleep", "state", "tremor"], spec=spec)
    model.fit(train, seed=0)                 # val is checked but not used: there's no early stopping
    predictions = model.predict(data)        # every diary hour, and every worn window

Doses are left out: in the prototype every dose feature tried (raw LEDD, LEDD relative to the patient's
usual dose, timing only) made these classifiers worse on new patients (EXPERIMENTS.md, Window
classifiers), because each lets a model learn the training patients' routines.

Training reads at most windows_per_hour windows from each answered hour (60, sampled with the seed), so
every hour counts about the same and the training set fits in memory. Class weights come from the
training hours (not windows), balanced as the transformer's are. A sample weight per hour (bias
mitigation) multiplies them and also weights the hours counted, so an hour of weight 0 is as if
it weren't there.

harnet10's features come from the shared cache (motor_causal_transformer/harnet.py, feature_cache), so
the transformer and these baselines compute them once. A saved model records harnet10's weights
fingerprint, and predict refuses other weights. The classifiers are saved with joblib, which is pickle:
load only models this pipeline wrote.
"""

import json
import numbers
from abc import abstractmethod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.core.data import SECONDS_PER_HOUR
from scripts.core.errors import ConfigError, ModelError, SchemaError
from scripts.core.model import Model, Predictions, WindowPredictions
from scripts.core.targets import HORIZONS, HOUR_INDEX, PARTICIPANT, get_target, parse_label
from scripts.models.motor_causal_transformer.harnet import check_spec, default_device, feature_cache
from scripts.models.motor_causal_transformer.loss import balanced_weights
from scripts.utils.logging_setup import get_logger

logger = get_logger(__name__)

CLASSIFIERS_FILE, INFO_FILE = "classifiers.joblib", "training.json"
BATCH = 50_000                     # windows scored at a time: about 200 MB of features


class WindowClassifier(Model):
    """The shared part of the window classifiers; a family only says which classifier (_classifier)."""

    needs = ("windows",)
    scores_windows = True

    @classmethod
    def default_params(cls):
        return {"windows_per_hour": 60, "class_weight_power": 1.0, "class_weight_cap": 10.0}

    def __init__(self, targets, horizons=tuple(HORIZONS), params=None, spec=None):
        super().__init__(targets, horizons, params, spec)
        _check_number(self.params, "windows_per_hour", whole=True, low=1)
        _check_number(self.params, "class_weight_power", low=0)
        _check_number(self.params, "class_weight_cap", low=1)
        self._check_params()
        self._classifiers = None       # {label: fitted pipeline, or None if training had one class}
        self._prior = None             # {label: class mix of the training hours}
        self._fingerprint = None       # harnet10's weights, once fitted
        self._result = None

    def _check_params(self):
        """A family checks its own settings here."""

    @abstractmethod
    def _classifier(self, seed):
        """A new, unfitted scikit-learn pipeline whose last step takes sample_weight."""

    @property
    def training_result(self):
        """How many windows and hours each label trained on (None before fitting)."""
        return json.loads(json.dumps(self._result)) if self._result is not None else None

    # ---- fit -----------------------------------------------------------------------------------

    def _fit(self, train, val, sample_weight, seed):
        check_spec(self.spec)
        cache = feature_cache(default_device())
        rng = np.random.default_rng(seed)
        hours = train.hours.reset_index(drop=True)
        labels = train.labels.reset_index(drop=True).reindex(columns=list(self.labels))
        hour_weight = np.ones(len(hours)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        answered = labels.notna().any(axis=1).to_numpy()

        x, hour_of_row = self._training_windows(train, hours, answered, cache, rng)
        y_hours = labels.to_numpy(dtype=float)
        classifiers, prior, result = {}, {}, {"windows": int(len(x)), "labels": {}}
        for j, label in enumerate(self.labels):
            classes = len(get_target(parse_label(label)[0]).classes)
            hour_y = y_hours[:, j]
            known = ~np.isnan(hour_y)
            # Weighted, so an hour of weight 0 counts as if it weren't there; +1 so no class is impossible.
            counts = 1 + np.bincount(hour_y[known].astype(int), weights=hour_weight[known], minlength=classes)[:classes]
            prior[label] = (counts / counts.sum()).tolist()
            y = hour_y[hour_of_row]
            has = ~np.isnan(y) & (hour_weight[hour_of_row] > 0)
            y = y[has].astype(int)
            result["labels"][label] = {"windows": int(has.sum()), "hours": int(known.sum()),
                                       "classes_seen": int(len(np.unique(y)))}
            if len(np.unique(y)) < 2:
                # Nothing to learn from no windows or one class: the label gets the training class mix everywhere.
                logger.warning("%s: %s has %s in training; it predicts the class mix", self.family, label,
                               "one class" if len(y) else "no windows", extra={"event": "label_not_learned",
                                                                               "label": label})
                classifiers[label] = None
                continue
            class_weight = balanced_weights(counts, self.params["class_weight_power"],
                                            self.params["class_weight_cap"]).numpy().astype(float)
            weight = class_weight[y] * hour_weight[hour_of_row][has]
            pipeline = self._classifier(seed)
            pipeline.fit(x[has], y, **{f"{pipeline.steps[-1][0]}__sample_weight": weight})
            classifiers[label] = pipeline
        self._classifiers, self._prior, self._result = classifiers, prior, result
        self._fingerprint = cache.harnet.fingerprint
        logger.info("%s trained on %d windows from %d hours", self.family, len(x), int(answered.sum()),
                    extra={"event": "window_classifier_fit", "windows": int(len(x)), "hours": int(answered.sum())})

    def _training_windows(self, train, hours, answered, cache, rng):
        """At most windows_per_hour worn windows of every answered hour: their features, and each one's hour row."""
        per_hour = self.params["windows_per_hour"]
        xs, rows = [], []
        for pid in train.participants:
            mine = np.flatnonzero((hours[PARTICIPANT] == pid).to_numpy() & answered)
            if not len(mine):
                continue
            windows = train.windows[pid]
            hour_row = _hour_rows(windows.time_s[windows.worn], hours[HOUR_INDEX].to_numpy()[mine], mine)
            in_hour = np.flatnonzero(hour_row >= 0)
            if not len(in_hour):
                continue
            # A random rank within each hour; keep the first per_hour. Sorted back, so reads go in order.
            order = np.lexsort((rng.random(len(in_hour)), hour_row[in_hour]))
            group = hour_row[in_hour][order]
            starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
            rank = np.arange(len(group)) - np.repeat(starts, np.diff(np.r_[starts, len(group)]))
            chosen = np.sort(in_hour[order][rank < per_hour])
            xs.append(np.asarray(cache.get(pid, windows)[chosen], dtype=np.float32))
            rows.append(hour_row[chosen])
        if not xs:
            raise SchemaError(f"{self.family}: the training data has no worn windows in answered hours")
        return np.concatenate(xs), np.concatenate(rows)

    # ---- predict -------------------------------------------------------------------------------

    def _predict(self, data):
        cache = feature_cache(default_device(), self._fingerprint)
        hours = data.hours.reset_index(drop=True)
        k = {label: len(get_target(parse_label(label)[0]).classes) for label in self.labels}
        sums = {label: np.zeros((len(hours), k[label])) for label in self.labels}
        counts = np.zeros(len(hours))
        windows_out = {}
        for pid in data.participants:
            windows = data.windows[pid]
            time_s = windows.time_s[windows.worn]
            features = cache.get(pid, windows)
            probs = {label: self._window_probs(label, features, k[label]) for label in self.labels}
            mine = np.flatnonzero((hours[PARTICIPANT] == pid).to_numpy())
            hour_row = _hour_rows(time_s, hours[HOUR_INDEX].to_numpy()[mine], mine)
            scored = hour_row >= 0                  # a window in an hour the table doesn't have is still scored
            counts += np.bincount(hour_row[scored], minlength=len(hours))
            for label in self.labels:
                np.add.at(sums[label], hour_row[scored], probs[label][scored])
            windows_out[pid] = WindowPredictions(np.asarray(time_s), probs)
        empty = counts == 0
        if empty.any():
            logger.info("%d of %d hours have no worn windows: given the training class mix", int(empty.sum()),
                        len(hours), extra={"event": "hours_without_windows", "hours": int(empty.sum())})
        hourly = {}
        for label in self.labels:
            p = sums[label] / np.maximum(counts, 1)[:, None]
            p[empty] = self._prior[label]
            hourly[label] = p / p.sum(axis=1, keepdims=True) if len(p) else p
        return Predictions(data.hours, hourly, windows_out)

    def _window_probs(self, label, features, classes):
        """Every worn window's class probabilities for one label, (windows, classes)."""
        pipeline = self._classifiers[label]
        out = np.empty((len(features), classes))
        if pipeline is None:
            out[:] = self._prior[label]
            return out
        known = pipeline.classes_.astype(int)          # classes training never saw stay at 0
        for start in range(0, len(features), BATCH):
            x = np.asarray(features[start:start + BATCH], dtype=np.float32)
            part = np.zeros((len(x), classes))
            part[:, known] = pipeline.predict_proba(x)
            out[start:start + len(x)] = part
        return out / out.sum(axis=1, keepdims=True) if len(out) else out

    # ---- save and load -------------------------------------------------------------------------

    def _save_weights(self, folder):
        joblib.dump(self._classifiers, Path(folder) / CLASSIFIERS_FILE)
        (Path(folder) / INFO_FILE).write_text(json.dumps(
            {"harnet10": self._fingerprint, "prior": self._prior, "result": self._result}, indent=1),
            encoding="utf-8")

    def _load_weights(self, folder):
        self._classifiers = joblib.load(Path(folder) / CLASSIFIERS_FILE)
        if set(self._classifiers) != set(self.labels):
            raise ModelError(f"{folder}: the saved classifiers don't match the model's labels")
        info = json.loads((Path(folder) / INFO_FILE).read_text(encoding="utf-8"))
        self._fingerprint, self._prior, self._result = info["harnet10"], info["prior"], info["result"]


class WindowLogistic(WindowClassifier):
    """Standardise each harnet10 feature, then multinomial logistic regression (L2, strength 1 / c)."""

    family = "window_logistic"

    @classmethod
    def default_params(cls):
        return {**super().default_params(), "c": 0.01, "max_iter": 500}

    def _check_params(self):
        _check_number(self.params, "c", low=0, strict=True)
        _check_number(self.params, "max_iter", whole=True, low=1)

    def _classifier(self, seed):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(C=self.params["c"], max_iter=self.params["max_iter"]))


class WindowBoosting(WindowClassifier):
    """Standardise, PCA to pca_components, then histogram gradient boosting (scikit-learn's)."""

    family = "window_boosting"

    @classmethod
    def default_params(cls):
        return {**super().default_params(), "pca_components": 64, "iterations": 200, "learning_rate": 0.05,
                "leaves": 15}

    def _check_params(self):
        _check_number(self.params, "pca_components", whole=True, low=1)
        _check_number(self.params, "iterations", whole=True, low=1)
        _check_number(self.params, "learning_rate", low=0, strict=True)
        _check_number(self.params, "leaves", whole=True, low=2)

    def _classifier(self, seed):
        from sklearn.decomposition import PCA
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        p = self.params
        return make_pipeline(StandardScaler(), PCA(p["pca_components"], random_state=seed),
                             HistGradientBoostingClassifier(max_iter=p["iterations"], learning_rate=p["learning_rate"],
                                                            max_leaf_nodes=p["leaves"], early_stopping=False,
                                                            random_state=seed))


def _hour_rows(time_s, hour_index, rows):
    """For each window, the row (of rows) of the hour it falls in, or -1 if that hour isn't one of them."""
    window_hour = np.floor(np.asarray(time_s, dtype=np.float64) / SECONDS_PER_HOUR).astype(np.int64)
    found = pd.Index(np.asarray(hour_index, dtype=np.int64)).get_indexer(window_hour)
    return np.where(found >= 0, np.asarray(rows)[np.maximum(found, 0)], -1)


def _check_number(params, name, whole=False, low=None, strict=False):
    value = params[name]
    kind = numbers.Integral if whole else numbers.Real
    bad = isinstance(value, bool) or not isinstance(value, kind) or not np.isfinite(value)
    if not bad and low is not None:
        bad = value <= low if strict else value < low
    if bad:
        what = "a whole number" if whole else "a number"
        bound = "" if low is None else f" {'above' if strict else 'of at least'} {low}"
        raise ConfigError(f"{name} must be {what}{bound}, not {value!r}")


__all__ = ["WindowBoosting", "WindowClassifier", "WindowLogistic"]
