"""Adds what train's scores leave out, writes metrics.json and the plots, and applies the validation gate.

Owner: P1 (component F).

train.py already scores every candidate's out-of-fold hours (core/metrics.py): cv/<label>/<metric>,
pooled and within patients, and _fold_sd. This step reads its outputs,

    artifacts/candidates.json           name, metrics, ... per candidate
    artifacts/predictions/<label>.csv   participant_id, hour_index, fold, seed, model, worn, y_true,
                                        then one column per class

and, for each candidate, on the same hours train scores (labelled, with a worn window, out of fold):

    cv/<label>/<question>_auc_lo, _hi   a 95% interval for the pooled AUC, from resampling patients
                                        (ARCHITECTURE.md, Evaluation protocol), averaged over seeds
    evaluation.json                     each question's AUC per fold and per patient
    roc.png                             the pooled ROC curve of every question, per label

It writes artifacts/evaluation/<candidate>/metrics.json (train's metrics plus the intervals, the
flat {name: value} the release gate reads), and sets each candidate's gates_passed: False when a
metric in validation_gate.min is missing or below its value. select.py then keeps only those that
passed (and the bias step can only turn a pass into a fail).
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from model_pipeline.core.metrics import QUESTIONS, auc
from model_pipeline.core.targets import PARTICIPANT, get_target, parse_label
from model_pipeline.utils.file_io import read_json, write_json
from model_pipeline.utils.logging_setup import get_logger

logger = get_logger(__name__)


def scored_rows(folder, name):
    """{label: rows} of one candidate's out-of-fold hours that count: labelled, worn, not the test patients."""
    out = {}
    for path in sorted(Path(folder).glob("*.csv")):
        rows = pd.read_csv(path)
        rows = rows[(rows["model"] == name) & (rows["fold"].astype(str) != "test")
                    & rows["worn"].astype(bool) & rows["y_true"].notna()]
        if len(rows):
            out[path.stem] = rows.reset_index(drop=True)
    return out


def questions(label, rows):
    """[(question, y, score)]: each broad question of the label's target, as yes/no and P(yes)."""
    target = parse_label(label)[0]
    classes = list(get_target(target).classes)
    y = rows["y_true"].to_numpy(int)
    return [(q, np.isin(y, c).astype(int), rows[[classes[i] for i in c]].sum(axis=1).to_numpy())
            for q, c in QUESTIONS[target].items()]


def interval(y, score, groups, resamples, seed, level):
    """A percentile interval for the pooled AUC, resampling patients with replacement. NaN if it can't be had."""
    rng = np.random.default_rng(seed)
    rows_of = {p: np.flatnonzero(groups == p) for p in np.unique(groups)}
    pids = list(rows_of)
    values = []
    for _ in range(resamples):
        idx = np.concatenate([rows_of[p] for p in rng.choice(pids, len(pids))])
        values.append(auc(y[idx], score[idx]))
    values = np.array([v for v in values if math.isfinite(v)])
    if not len(values):
        return math.nan, math.nan
    tail = 100 * (1 - level) / 2
    return float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))


def by(values, y, score):
    """{group: AUC} for each value of values (a fold or a patient), NaN where it has one class only."""
    values = np.asarray(values).astype(str)
    return {g: auc(y[values == g], score[values == g]) for g in sorted(set(values))}


def evaluate_candidate(rows_by_label, cfg):
    """(extra metrics, details) for one candidate: the intervals, and AUCs per fold and per patient."""
    ev, seed = cfg["evaluation"], cfg["data"]["random_seed"]
    metrics, details = {}, {}
    for label, rows in rows_by_label.items():
        for question, _, _ in questions(label, rows):
            bounds = []
            for _, part in rows.groupby("seed"):
                _, y, score = next(q for q in questions(label, part) if q[0] == question)
                bounds.append(interval(y, score, part[PARTICIPANT].to_numpy(), ev["bootstrap_resamples"], seed,
                                       ev["interval"]))
            lo, hi = (float(np.nanmean(b)) if not np.isnan(b).all() else math.nan for b in zip(*bounds))
            metrics[f"cv/{label}/{question}_auc_lo"], metrics[f"cv/{label}/{question}_auc_hi"] = lo, hi
        first = rows[rows["seed"] == rows["seed"].min()]
        for question, y, score in questions(label, first):
            details[f"{label}/{question}_auc"] = {"folds": by(first["fold"], y, score),
                                                  "patients": by(first[PARTICIPANT], y, score)}
    return metrics, details


def gate_failures(metrics, minimums):
    """The validation gate's failures, as readable messages. A missing or NaN metric fails."""
    failures = []
    for name, lowest in (minimums or {}).items():
        value = metrics.get(name, math.nan)
        if not (isinstance(value, (int, float)) and math.isfinite(value)):
            failures.append(f"{name}: missing")
        elif value < lowest:
            failures.append(f"{name}: {value:.3f} < {lowest}")
    return failures


def plot(rows_by_label, metrics, path):
    """The pooled ROC curve of each question, one panel per label, for the first seed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    labels = sorted(rows_by_label)
    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 4), squeeze=False)
    for ax, label in zip(axes[0], labels):
        rows = rows_by_label[label]
        for question, y, score in questions(label, rows[rows["seed"] == rows["seed"].min()]):
            if len(set(y)) == 2:
                fpr, tpr, _ = roc_curve(y, score)
                ax.plot(fpr, tpr, label=f"{question} {metrics.get(f'cv/{label}/{question}_auc', math.nan):.2f}")
        ax.plot([0, 1], [0, 1], color="grey", linestyle=":")
        ax.set(title=label, xlabel="false positive rate", ylabel="true positive rate")
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


class EvaluateModel:
    """Step F: see the top of this file."""

    def __init__(self, cfg, tracker):
        self.cfg, self.tracker = cfg, tracker
        self.artifacts = Path(cfg["resolved_paths"]["artifacts_dir"])

    def run(self, candidates=None):
        """The candidates, each with its metrics extended and gates_passed set. Reads candidates.json if None."""
        candidates = candidates if candidates is not None else read_json(self.artifacts / "candidates.json")
        minimums = self.cfg["validation_gate"]["min"]
        with self.tracker.run("evaluate", tags={"owner": "P1"}):
            for c in candidates:
                rows = scored_rows(self.artifacts / "predictions", c["name"])
                extra, details = evaluate_candidate(rows, self.cfg)
                c["metrics"] = {**c["metrics"], **extra}
                folder = self.artifacts / "evaluation" / c["name"]
                write_json(_finite(c["metrics"]), folder / "metrics.json")
                write_json(_finite(details), folder / "evaluation.json")
                if rows:
                    plot(rows, c["metrics"], folder / "roc.png")
                with self.tracker.run(c["name"], tags={"run_id": str(c.get("run_id"))}):
                    self.tracker.log_metrics({n: v for n, v in extra.items() if math.isfinite(v)})
                    self.tracker.log_artifact(folder, "evaluation")
                failures = gate_failures(c["metrics"], minimums)
                c["gates_passed"] = not failures
                c["validation_failures"] = failures
                logger.info("%s: validation gate %s%s", c["name"], "failed: " if failures else "passed",
                            "; ".join(failures))
        return candidates


def _finite(value):
    """NaN as None, all the way down, so the JSON is valid."""
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
