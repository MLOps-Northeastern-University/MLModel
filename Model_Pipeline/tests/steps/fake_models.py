"""A small model family and data for the bias and explain tests: no PyTorch, no harnet10.

HourlyLogistic follows the real Model contract (core/model.py), so the steps see exactly what they
would from a real family. It reads three input groups, so explain has something to attribute:
the 7 hourly features, whether the hour had a dose (events), and DBS (patients).

biased_dataset() makes data with one of two kinds of bias, each giving a real recall gap:

    signal      movement shows OFF clearly for women and faintly for men. Reweighing can't fix this:
                OFF is as common in both, so there is nothing to rebalance; the information is weaker.
    imbalance   OFF is rare in DBS patients and the model reads DBS, so it learns "DBS: probably not
                OFF" and misses DBS patients' OFF hours. Reweighing should fix this.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from model_pipeline.core.data import Dataset, WindowSpec
from model_pipeline.core.model import Model, Predictions
from model_pipeline.core.targets import HOUR_INDEX, PARTICIPANT, build_labels, get_target, parse_label
from model_pipeline.data.folds import Split

FEATURES = ["move_mean", "move_std", "move_p95", "still_fraction", "tremor_mean", "tremor_p95", "tremor_ratio"]
SPEC = WindowSpec(1, 10, 30, 3)
PIDS = [f"P{i:02d}" for i in range(1, 13)]
SPLIT = Split(["P11", "P12"], [PIDS[0:3], PIDS[3:6], PIDS[6:10]], seed=0)


class HourlyLogistic(Model):
    """One logistic regression per label on [features, dosed this hour, DBS]."""

    family = "fake_hourly"
    needs = ("features", "events", "patients")

    @classmethod
    def default_params(cls):
        return {"C": 1.0}

    def _design(self, data):
        hours = data.hours.reset_index(drop=True)
        doses = data.events[data.events["kind"] == "dose"]
        dosed = set(zip(doses[PARTICIPANT], (doses["time_s"] // 3600).astype(int)))
        dbs = dict(zip(data.patients[PARTICIPANT], data.patients["dbs"].astype(bool)))
        return np.column_stack([
            data.features.reset_index(drop=True).to_numpy(float),
            [float((p, int(h)) in dosed) for p, h in zip(hours[PARTICIPANT], hours[HOUR_INDEX])],
            [float(dbs[p]) for p in hours[PARTICIPANT]],
        ])

    def _fit(self, train, val, sample_weight, seed):
        X = self._design(train)
        self._fits = {}
        for label in self.labels:
            k = len(get_target(parse_label(label)[0]).classes)
            y = train.labels[label].to_numpy(float)
            known = ~np.isnan(y)
            seen = np.unique(y[known]).astype(int)
            if len(seen) < 2:
                prior = np.full(k, 1.0 / k)
                self._fits[label] = ("constant", prior)
                continue
            weights = None if sample_weight is None else sample_weight[known]
            fit = LogisticRegression(C=self.params["C"], max_iter=2000, random_state=seed)
            fit.fit(X[known], y[known].astype(int), sample_weight=weights)
            self._fits[label] = ("logistic", fit)

    def _predict(self, data):
        X = self._design(data)
        probs = {}
        for label in self.labels:
            k = len(get_target(parse_label(label)[0]).classes)
            kind, fit = self._fits[label]
            out = np.zeros((len(X), k))
            if kind == "constant":
                out[:] = fit
            else:
                out[:, fit.classes_] = fit.predict_proba(X)
            probs[label] = out
        return Predictions(hours=data.hours, probs=probs)

    def _save_weights(self, folder):
        joblib.dump(self._fits, folder / "fits.joblib")

    def _load_weights(self, folder):
        self._fits = joblib.load(folder / "fits.joblib")


def biased_dataset(hours=40, seed=0, kind="signal"):
    """12 patients (sex alternating, two age groups, DBS by pairs), `hours` diary hours each.

    move_mean is higher in OFF hours, in noise of 1; one hour in ten is asleep (move_mean lower); each
    patient has a dose at hour 2. kind is "signal" or "imbalance" (see the top of this file).
    """
    rng = np.random.default_rng(seed)
    dbs = [(i // 2) % 2 == 1 for i in range(len(PIDS))]
    hour_rows, feature_rows, events = [], [], []
    for i, pid in enumerate(PIDS):
        sex = "female" if i % 2 == 0 else "male"
        if kind == "signal":
            strength, p_off = (1.5 if sex == "female" else 0.25), 0.4
        else:
            strength, p_off = 1.5, (0.15 if dbs[i] else 0.6)
        for h in range(hours):
            asleep = rng.random() < 0.1
            off = (not asleep) and rng.random() < p_off
            score = np.nan if asleep else float(rng.choice([-2, -1])) if off else float(rng.choice([0, 0, 1]))
            hour_rows.append({PARTICIPANT: pid, HOUR_INDEX: h, "kinesia_text": "Sleep" if asleep else "Answer",
                              "kinesia_score": score, "tremor_score": np.nan if asleep else float(rng.integers(0, 3)),
                              "sex": sex, "age_group": "<65" if i < 6 else "65+"})
            noise = rng.normal(0, 1, len(FEATURES))
            noise[0] += strength * (1.0 if off else 0.0) - (2.0 if asleep else 0.0)
            feature_rows.append(noise)
        events.append({PARTICIPANT: pid, "time_s": 2 * 3600 + 60.0, "kind": "dose", "ledd": 100.0,
                       "source": "on_time", "kinesia_score": np.nan, "tremor_score": np.nan})
    table = pd.DataFrame(hour_rows)
    return Dataset(
        spec=SPEC,
        hours=table[[PARTICIPANT, HOUR_INDEX, "sex", "age_group"]],
        labels=build_labels(table, ["sleep", "state", "tremor"], ["now"]),
        features=pd.DataFrame(feature_rows, columns=FEATURES),
        events=pd.DataFrame(events),
        patients=pd.DataFrame({PARTICIPANT: PIDS, "more_affected_side": [i % 3 == 0 for i in range(12)],
                               "dbs": dbs}),
    )


def pipeline_config(tmp_path, **bias):
    """The config sections train, bias and explain read, with every output under tmp_path."""
    return {
        "windows": {"spec_version": 1, "seconds": 10, "rate_hz": 30, "axes": 3},
        "resolved_paths": {"artifacts_dir": tmp_path},
        "models": {"targets": ["sleep", "state", "tremor"], "horizons": ["now"],
                   "candidates": [{"name": "fake_hourly", "family": "fake_hourly"}]},
        "train": {"seeds": [0], "score_test": False, "leakage_check": {"candidates": []}},
        "selection": {"metric": "cv/state_now/off_auc"},
        "alerts": {"email_enabled": False, "slack_enabled": False},
        "bias": {"slice_columns": ["sex", "age_group", "dbs"], "max_recall_gap": 0.10, "min_slice_rows": 20,
                 "min_slice_participants": 2, "threshold": 0.5,
                 "checks": [{"label": "state_now", "metric": "off_recall"},
                            {"label": "sleep_now", "metric": "asleep_auc", "max_gap": 0.5}],
                 "mitigation": {"enabled": True, "candidates": [], "power": 1.0, "max_weight": 5.0}, **bias},
        "explain": {"patients": "development", "max_participants": 4, "repeats": 2, "seed": 0, "candidates": [],
                    "outputs": [{"label": "state_now", "question": "off"},
                                {"label": "sleep_now", "question": "asleep"}]},
    }
