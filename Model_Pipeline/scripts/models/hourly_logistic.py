"""The hourly baseline: logistic regression on each diary hour's 7 sensor features, dose timing and time of day.

Owner: P1.

The features are data/baseline_loader.py's: the W_ features relative to the patient's own past,
hours since the last dose and recent LEDD, and the hour of day as sin and cos. One scaler +
LogisticRegression per label (sleep_now, state_now, tremor_now, and their next_hour versions).

The state is fitted as 3 groups, OFF (score below 0), ON (0) and dyskinetic (above 0), since the
severe scores have too few hours to learn apart. Each group's probability is then spread over its
scores in the proportions training saw, so the output still has the target's 7 classes.

    params:
      c: 1.0                    # inverse regularisation strength
      class_weight: balanced    # or null: every hour weighs the same
      max_iter: 1000
      features: {...}           # overrides of baseline_loader.DEFAULTS
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from scripts.core.errors import ConfigError, ModelError
from scripts.core.model import Model, Predictions
from scripts.core.targets import HORIZONS, STATE_GROUPS, get_target, parse_label
from scripts.data import baseline_loader as bl
from scripts.utils.logging_setup import get_logger

logger = get_logger(__name__)

CLASSIFIERS_FILE, INFO_FILE = "classifiers.joblib", "training.json"
GROUP_OF_STATE = np.empty(7, dtype=int)
for _g, (_name, _scores) in enumerate(STATE_GROUPS.items()):
    GROUP_OF_STATE[list(_scores)] = _g


class HourlyLogistic(Model):
    family = "hourly_logistic"
    needs = ("features", "events")

    @classmethod
    def default_params(cls):
        return {"c": 1.0, "class_weight": "balanced", "max_iter": 1000, "features": {}}

    def __init__(self, targets, horizons=tuple(HORIZONS), params=None, spec=None):
        super().__init__(targets, horizons, params, spec)
        if not (isinstance(self.params["c"], (int, float)) and self.params["c"] > 0):
            raise ConfigError(f"{self.family}: c must be a number above 0")
        if self.params["class_weight"] not in ("balanced", None):
            raise ConfigError(f"{self.family}: class_weight must be balanced or null")
        try:
            self._settings = bl.settings(self.params["features"])
        except ValueError as e:
            raise ConfigError(f"{self.family}: {e}") from None

    # ---- the features --------------------------------------------------------------------------

    def _table(self, data):
        """data.hours with the W_ columns from data.features (the loader's names, without the prefix)."""
        hours = data.hours.copy()
        for name in bl.FEATURE_NAMES:
            hours[f"W_{name}"] = data.features[name].to_numpy()
        # The data pipeline's hours table has these; a table without them (a live feed, tests) gets the
        # hour of day from hour_index, and every hour with all 7 features counts as good.
        if "hour_start" not in hours:
            hours["hour_start"] = hours["hour_index"] % 24
        if "W_status" not in hours:
            complete = data.features[bl.FEATURE_NAMES].notna().all(axis=1).to_numpy()
            hours["W_status"] = np.where(complete, "ok", "no_data")
        for column in ("W_valid_fraction", "worn_fraction"):
            if column not in hours:
                hours[column] = 1.0
        return hours

    def _x(self, data):
        hours = self._table(data)
        events = data.events if data.events is not None else hours.iloc[:0]
        return bl.feature_matrix(hours, events, self._settings, self._cohort).to_numpy(dtype=float)

    # ---- fit -----------------------------------------------------------------------------------

    def _fit(self, train, val, sample_weight, seed):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self._cohort = bl.cohort_stats(self._table(train), self._settings)
        x = self._x(train)
        labels = train.labels.reset_index(drop=True).reindex(columns=list(self.labels))
        weight = np.ones(len(x)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self._classifiers, self._prior, self._spread, self._result = {}, {}, {}, {}
        for label in self.labels:
            target = parse_label(label)[0]
            k = len(get_target(target).classes)
            y = labels[label].to_numpy(dtype=float)
            has = ~np.isnan(y) & (weight > 0)
            y = y[has].astype(int)
            counts = 1 + np.bincount(y, minlength=k)[:k]          # +1: no class impossible
            self._prior[label] = (counts / counts.sum()).tolist()
            if target == "state":
                # Within each group, the scores' share of that group's training hours.
                self._spread[label] = (counts / np.bincount(GROUP_OF_STATE, weights=counts)[GROUP_OF_STATE]).tolist()
                y = GROUP_OF_STATE[y]
            self._result[label] = {"hours": int(has.sum()), "classes_seen": int(len(np.unique(y)))}
            if len(np.unique(y)) < 2:
                logger.warning("%s: %s has %s in training; it predicts the class mix", self.family, label,
                               "one class" if len(y) else "no hours", extra={"event": "label_not_learned",
                                                                             "label": label})
                self._classifiers[label] = None
                continue
            pipeline = make_pipeline(StandardScaler(), LogisticRegression(
                C=self.params["c"], class_weight=self.params["class_weight"], max_iter=self.params["max_iter"],
                random_state=seed))
            pipeline.fit(x[has], y, logisticregression__sample_weight=weight[has])
            self._classifiers[label] = pipeline
        logger.info("%s trained on %d hours", self.family, len(x), extra={"event": "hourly_logistic_fit",
                                                                          "labels": self._result})

    # ---- predict -------------------------------------------------------------------------------

    def _predict(self, data):
        x = self._x(data)
        probs = {}
        for label in self.labels:
            target = parse_label(label)[0]
            k = len(get_target(target).classes)
            pipeline = self._classifiers[label]
            if pipeline is None:
                probs[label] = np.tile(self._prior[label], (len(x), 1))
                continue
            width = len(STATE_GROUPS) if target == "state" else k
            p = np.zeros((len(x), width))
            p[:, pipeline.classes_.astype(int)] = pipeline.predict_proba(x)
            if target == "state":
                p = p[:, GROUP_OF_STATE] * np.asarray(self._spread[label])   # group -> its scores
            probs[label] = p / p.sum(axis=1, keepdims=True) if len(p) else p
        return Predictions(data.hours, probs)

    # ---- save and load -------------------------------------------------------------------------

    def _save_weights(self, folder):
        import sklearn
        folder = Path(folder)
        joblib.dump(self._classifiers, folder / CLASSIFIERS_FILE)
        (folder / INFO_FILE).write_text(json.dumps({
            "cohort": [v.tolist() for v in self._cohort], "prior": self._prior, "spread": self._spread,
            "result": self._result, "scikit_learn": sklearn.__version__,
            "classifiers_sha256": _sha256(folder / CLASSIFIERS_FILE)}, indent=1), encoding="utf-8")

    def _load_weights(self, folder):
        """Check the pickle's SHA-256 before unpickling it: unpickling runs code."""
        folder = Path(folder)
        info = json.loads((folder / INFO_FILE).read_text(encoding="utf-8"))
        if _sha256(folder / CLASSIFIERS_FILE) != info.get("classifiers_sha256"):
            raise ModelError(f"{folder}: {CLASSIFIERS_FILE} isn't the file this model saved (its SHA-256 differs)")
        self._classifiers = joblib.load(folder / CLASSIFIERS_FILE)
        if set(self._classifiers) != set(self.labels):
            raise ModelError(f"{folder}: the saved classifiers don't match the model's labels")
        self._cohort = tuple(np.asarray(v) for v in info["cohort"])
        self._prior, self._spread, self._result = info["prior"], info["spread"], info["result"]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
