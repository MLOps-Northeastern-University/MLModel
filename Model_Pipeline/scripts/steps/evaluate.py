"""Scores the cross-validation predictions and applies the validation gate.

Owner: P1 (component F).

Predictions are one row per hour and label: participant_id, hour_index, fold,
label (such as state_now), an optional seed, and one probability column per
class of the label's target, p_<class> (prediction_columns). They're checked
against the diary labels in the Dataset; hours with no label are left out.

Each target is scored on events, a group of its classes (EVENTS): OFF is
P(score -3 to -1), any tremor is P(tremor score 1 or 2). For each label and event:

    cv/<label>/<event>             AUC inside each fold, averaged over folds (the gate)
    cv/<label>/<event>_no_dbs      the same, on participants without DBS
    cv/<label>/pooled_<event>      every test hour scored together, with _lo and _hi,
                                   a 95% interval from resampling participants
    cv/<label>/tracking_spearman   per participant, the predicted score against
                                   the diary's, averaged over those whose diary varies

A fold where an event's AUC can't be computed (one class only) is left out of
the average, and the number of folds used is reported with _folds. With several
seeds, each metric is the mean over seeds, with its spread as _seed_std.

Writes artifacts/metrics.json (the flat {name: value} the release gate reads),
artifacts/evaluation.json (per fold and per participant) and the plots, and
raises GateError when a metric in validation_gate.min is missing or below it.
"""

import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from scripts.core.errors import GateError, SchemaError
from scripts.core.targets import ASLEEP, HOUR_INDEX, PARTICIPANT, STATE_GROUPS, get_target
from scripts.utils.file_io import write_json
from scripts.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Each target's events: metric name -> the classes whose probabilities add up to it.
EVENTS = {
    "state": {"off_auc": STATE_GROUPS["off"], "dyskinetic_auc": STATE_GROUPS["dyskinetic"],
              "asleep_auc": STATE_GROUPS["asleep"]},
    "tremor": {"auc": (1, 2)},
}
KEYS = (PARTICIPANT, HOUR_INDEX)


def target_of(label):
    """The target a label belongs to: state_next_hour -> state."""
    return label.split("_", 1)[0]


def prediction_columns(target):
    """The probability columns a target's predictions need, in class order."""
    return [f"p_{c}" for c in get_target(target).classes]


def attach_truth(predictions, dataset):
    """The predictions with y_true (the class index from the diary) and dbs, labelled hours only.

    Raises:
        SchemaError: A column is missing, probabilities don't add up to 1, or a
            prediction is for an hour or label the Dataset doesn't have.
    """
    missing = [c for c in KEYS + ("fold", "label") if c not in predictions.columns]
    if missing:
        raise SchemaError(f"predictions need column(s) {missing}")
    parts = []
    for label, rows in predictions.groupby("label", sort=True):
        if label not in dataset.labels.columns:
            raise SchemaError(f"predictions for {label!r}, which the dataset has no labels for")
        probs = prediction_columns(target_of(label))
        absent = [c for c in probs if c not in rows.columns]
        if absent:
            raise SchemaError(f"{label} predictions need column(s) {absent}")
        p = rows[probs].to_numpy(float)
        if not (np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(axis=1), 1, atol=1e-3)):
            raise SchemaError(f"{label}: each row's probabilities must be 0 or more and add up to 1")
        truth = dataset.hours[list(KEYS)].assign(y_true=dataset.labels[label].to_numpy())
        rows = rows.merge(truth, on=list(KEYS), how="left", indicator=True)
        if (rows["_merge"] == "left_only").any():
            raise SchemaError(f"{label}: predictions for hours that aren't in the dataset")
        parts.append(rows.drop(columns="_merge").dropna(subset=["y_true"]))
    out = pd.concat(parts, ignore_index=True)
    dbs = dataset.patients.set_index(PARTICIPANT)["dbs"] if dataset.patients is not None else None
    out["dbs"] = out[PARTICIPANT].map(dbs) if dbs is not None else np.nan
    return out


def auc(y, score):
    """ROC AUC, or NaN when y has only one class."""
    return float(roc_auc_score(y, score)) if len(set(y)) == 2 else math.nan


def expected_score(rows, target):
    """The predicted score: expected tremor score, or expected kinesia score on waking hours (else NaN)."""
    p = rows[prediction_columns(target)].to_numpy(float)
    if target == "tremor":
        return p @ np.arange(p.shape[1])
    waking = p[:, :ASLEEP]
    return np.where(rows["y_true"] < ASLEEP, waking @ np.arange(-3, 4) / waking.sum(axis=1).clip(1e-12), np.nan)


def tracking(rows, target):
    """Spearman of predicted against diary score per participant, for those whose diary varies."""
    rows = rows.assign(pred=expected_score(rows, target)).dropna(subset=["pred"])
    out = {}
    for pid, r in rows.groupby(PARTICIPANT):
        if r["y_true"].nunique() > 1 and r["pred"].nunique() > 1:
            out[pid] = float(r["pred"].corr(r["y_true"], method="spearman"))
    return out


def pooled_interval(y, score, groups, resamples, seed, level=0.95):
    """A percentile interval for the pooled AUC, resampling participants with replacement."""
    rng = np.random.default_rng(seed)
    pids = np.unique(groups)
    rows_of = {p: np.flatnonzero(groups == p) for p in pids}
    values = []
    for _ in range(resamples):
        idx = np.concatenate([rows_of[p] for p in rng.choice(pids, len(pids))])
        values.append(auc(y[idx], score[idx]))
    values = np.array(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    tail = 100 * (1 - level) / 2
    return float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))


def score_seed(rows, cfg):
    """The flat metrics and the per-fold and per-participant details for one seed's predictions."""
    ev = cfg["evaluation"]
    metrics, details = {}, {}
    for label, r in rows.groupby("label", sort=True):
        target = target_of(label)
        for event, classes in EVENTS[target].items():
            y = r["y_true"].isin(classes).to_numpy(int)
            score = r[[prediction_columns(target)[c] for c in classes]].sum(axis=1).to_numpy()
            folds = {str(f): auc(y[m], score[m]) for f, m in _masks(r["fold"])}
            no_dbs = (r["dbs"] == False).to_numpy()  # noqa: E712 (NaN is neither)
            folds_no_dbs = {str(f): auc(y[m & no_dbs], score[m & no_dbs]) for f, m in _masks(r["fold"])}
            people = {str(p): auc(y[m], score[m]) for p, m in _masks(r[PARTICIPANT])}
            lo, hi = pooled_interval(y, score, r[PARTICIPANT].to_numpy(), ev["bootstrap_resamples"],
                                     cfg["data"]["random_seed"], ev["interval"])
            base = f"cv/{label}/{event}"
            metrics.update({
                base: _mean(folds), f"{base}_folds": _count(folds),
                f"{base}_no_dbs": _mean(folds_no_dbs), f"{base}_no_dbs_folds": _count(folds_no_dbs),
                f"cv/{label}/pooled_{event}": auc(y, score),
                f"cv/{label}/pooled_{event}_lo": lo, f"cv/{label}/pooled_{event}_hi": hi,
            })
            details[f"{label}/{event}"] = {"folds": folds, "folds_no_dbs": folds_no_dbs, "participants": people}
        spearman = tracking(r, target)
        metrics[f"cv/{label}/tracking_spearman"] = _mean(spearman)
        metrics[f"cv/{label}/tracking_participants"] = len(spearman)
        details[f"{label}/tracking_spearman"] = spearman
    return metrics, details


def score(predictions, dataset, cfg):
    """(metrics, details) for the predictions: means over seeds, with _seed_std when there are several."""
    rows = attach_truth(predictions, dataset)
    seeds = rows.groupby("seed") if "seed" in rows.columns else [(None, rows)]
    results = {str(s): score_seed(r, cfg) for s, r in seeds}
    names = list(next(iter(results.values()))[0])
    table = pd.DataFrame([m for m, _ in results.values()], columns=names)
    metrics = table.mean().to_dict()
    if len(results) > 1:
        metrics.update({f"{n}_seed_std": float(table[n].std()) for n in names if not n.endswith(("_folds",
                                                                                                "_participants"))})
    details = {"seeds": {s: d for s, (_, d) in results.items()}}
    return metrics, details


def gate_failures(metrics, minimums):
    """The validation gate's failures, as readable messages. A missing or NaN metric fails."""
    failures = []
    for name, lowest in (minimums or {}).items():
        value = metrics.get(name, math.nan)
        if not math.isfinite(value) or value < lowest:
            failures.append(f"{name}: {value:.3f} < {lowest}" if math.isfinite(value) else f"{name}: missing")
    return failures


def plot(rows, metrics, folder):
    """roc.png: the pooled ROC curve of each label's events, side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = sorted(rows["label"].unique())
    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 4), squeeze=False)
    for ax, label in zip(axes[0], labels):
        r, target = rows[rows["label"] == label], target_of(label)
        for event, classes in EVENTS[target].items():
            y = r["y_true"].isin(classes).to_numpy(int)
            if len(set(y)) == 2:
                s = r[[prediction_columns(target)[c] for c in classes]].sum(axis=1)
                fpr, tpr, _ = roc_curve(y, s)
                ax.plot(fpr, tpr, label=f"{event} {metrics[f'cv/{label}/pooled_{event}']:.2f}")
        ax.plot([0, 1], [0, 1], color="grey", linestyle=":")
        ax.set(title=label, xlabel="false positive rate", ylabel="true positive rate")
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "roc.png", dpi=120)
    plt.close(fig)


class EvaluateModel:
    """Scores a candidate's cross-validation predictions and applies the validation gate."""

    def __init__(self, cfg, tracker):
        self.cfg = cfg
        self.tracker = tracker
        self.folder = cfg["resolved_paths"]["artifacts_dir"]

    def run(self, predictions, dataset):
        """The metrics. Raises GateError when one in validation_gate.min is missing or too low."""
        with self.tracker.run("evaluate", tags={"owner": "P1"}):
            metrics, details = score(predictions, dataset, self.cfg)
            self.tracker.log_metrics({n: v for n, v in metrics.items() if math.isfinite(v)})
            write_json({n: _finite(v) for n, v in metrics.items()}, self.folder / "metrics.json")
            write_json(_finite(details), self.folder / "evaluation.json")
            plot(attach_truth(predictions, dataset), metrics, self.folder)
            for name in ("metrics.json", "evaluation.json", "roc.png"):
                self.tracker.log_artifact(self.folder / name, "evaluation")
            failures = gate_failures(metrics, self.cfg["validation_gate"]["min"])
            self.tracker.set_tags({"validation_gate": "failed" if failures else "passed"})
            logger.info("off_auc %.3f over %d folds, tremor auc %.3f", metrics.get("cv/state_now/off_auc", math.nan),
                        metrics.get("cv/state_now/off_auc_folds", 0), metrics.get("cv/tremor_now/auc", math.nan))
            if failures:
                raise GateError("validation", "; ".join(failures))
            return metrics


def _masks(values):
    values = np.asarray(values)
    return [(v, values == v) for v in sorted(pd.unique(values), key=str)]


def _mean(scores):
    finite = [v for v in scores.values() if math.isfinite(v)]
    return float(np.mean(finite)) if finite else math.nan


def _count(scores):
    return sum(math.isfinite(v) for v in scores.values())


def _finite(value):
    """NaN as None, all the way down, so the JSON is valid."""
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
