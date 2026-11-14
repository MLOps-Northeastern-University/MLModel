"""The scores every step reports per label, each pooled over all hours and within patients.

Owner: P2, with P1 (whose evaluate step reports them).

Pooled and within answer different questions, so both are reported:

    pooled   every hour of every patient in one pile. It also rewards telling patients apart, so a
             model can score well by learning who is usually OFF.
    within   each patient's own score, from their hours only, averaged over the patients (for the
             AUCs, accuracy, F1 and correlation, those with at least two different answers; the
             error, score_mae_within, uses every patient). Only how well the model follows one patient through the
             day, which is what the dashboard shows (MODEL.md, "How OFF is defined and counted").

A metric name ends in _within for the within version; without it, it's pooled. Names follow the
tracker's scheme, {split}/{label}/{metric}, such as cv/state_now/off_auc_within.

The broad questions, each a sum of class probabilities (core/targets.py), scored by ROC AUC:

    sleep    asleep                  P(asleep)
    state    off, on, dyskinetic     P(-3..-1), P(0), P(+1..+3), each given awake
    tremor   any_tremor              P(1) + P(2)

    <question>_auc, <question>_auc_within, and for the state's OFF also off_auc_no_dbs (pooled,
    patients without DBS only: there are too few of them for a within average)

The full classes, from the most likely class per hour:

    balanced_accuracy   the mean of the classes' recalls, so rare classes count as much as common ones
    macro_f1            the mean of the classes' F1, over the classes in the answers or the predictions
    recall_<class>, precision_<class>   pooled only: per patient there are too few hours of each class.
                        NaN for a class with no answers (recall) or never predicted (precision), so a
                        class the model never predicts shows up as missing, not as a number

The diary score, for the targets that have one (state -3..+3, tremor 0..2), from the expected score
(sum of score x probability):

    score_spearman      the rank correlation with the diary score: does the model's score rise and fall
                        with the patient's? The order counts, so -1 for a true -3 is closer than +1
    score_mae           the mean absolute error, in score points

confusion() gives the counts behind the class metrics, for evaluate's plots.

Only numpy, so core stays light. Each is tested equal to scikit-learn's or SciPy's version.
"""

import math
from typing import Dict, Mapping, Optional

import numpy as np

from scripts.core.errors import ConfigError
from scripts.core.targets import ASLEEP, STATE_GROUPS, get_target, parse_label

# Scores where lower is better (an error); every other score is better higher. Tuning, selection and
# the release gate all rank by this, so a metric is never maximised by mistake.
LOWER_IS_BETTER = ("score_mae", "score_mae_within")


def higher_is_better(name: str) -> bool:
    """Whether a bigger value is better, for a score or a tracker name such as cv/state_now/score_mae.

    A _fold_sd (how much a score varies between folds) is better smaller, too.
    """
    metric = str(name).rsplit("/", 1)[-1]
    if metric.endswith("_fold_sd"):
        return False
    return metric not in LOWER_IS_BETTER


# target -> {question: the classes that mean "yes"}
QUESTIONS = {
    "sleep": {"asleep": (ASLEEP,)},
    "state": {"off": STATE_GROUPS["off"], "on": STATE_GROUPS["on"], "dyskinetic": STATE_GROUPS["dyskinetic"]},
    "tremor": {"any_tremor": (1, 2)},
}


# ---- the single scores --------------------------------------------------------------------------

def auc(y, score) -> float:
    """The ROC AUC of score for the yes/no y; NaN unless y has both answers.

    The Mann-Whitney statistic with average ranks for ties, which equals scikit-learn's roc_auc_score.
    """
    y, score = _pair(y, score, "auc")
    y = y.astype(bool)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = _average_ranks(score)
    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def spearman(a, b) -> float:
    """The rank correlation of a and b (ties share their average rank); NaN if either is constant."""
    a, b = _pair(a, b, "spearman")
    if len(a) < 2:
        return math.nan
    ra, rb = _average_ranks(a) - (len(a) + 1) / 2, _average_ranks(b) - (len(b) + 1) / 2
    denominator = math.sqrt(float((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / denominator) if denominator > 0 else math.nan


def mae(a, b) -> float:
    """The mean absolute difference; NaN for no values."""
    a, b = _pair(a, b, "mae")
    return float(np.abs(a - b).mean()) if len(a) else math.nan


def confusion(y, predicted, n_classes) -> np.ndarray:
    """(n_classes, n_classes) counts: row = the answer, column = the most likely class."""
    y, predicted = _pair(y, predicted, "confusion")
    out = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(out, (y.astype(int), predicted.astype(int)), 1)
    return out


def balanced_accuracy(y, predicted, n_classes) -> float:
    """The mean recall over the classes that appear in y, as scikit-learn's balanced_accuracy_score."""
    m = confusion(y, predicted, n_classes)
    present = m.sum(axis=1) > 0
    if not present.any():
        return math.nan
    return float((np.diag(m)[present] / m.sum(axis=1)[present]).mean())


def macro_f1(y, predicted, n_classes) -> float:
    """The mean F1 over the classes in y or predicted, as scikit-learn's f1_score(average="macro")."""
    m = confusion(y, predicted, n_classes)
    used = (m.sum(axis=0) + m.sum(axis=1)) > 0
    if not used.any():
        return math.nan
    tp = np.diag(m).astype(float)
    denominator = m.sum(axis=0) + m.sum(axis=1)           # 2TP + FP + FN
    f1 = np.where(denominator > 0, 2 * tp / np.maximum(denominator, 1), 0.0)
    return float(f1[used].mean())


def within(groups, function, *arrays, min_classes=None) -> float:
    """function(*arrays) for each group (patient) on its own rows, averaged, ignoring NaN.

    min_classes: skip groups whose first array has fewer different values (e.g. 2 for an AUC).
    """
    groups = np.asarray(groups)
    arrays = [np.asarray(a) for a in arrays]
    values = []
    for g in np.unique(groups):
        rows = groups == g
        if min_classes is not None and len(np.unique(arrays[0][rows])) < min_classes:
            continue
        values.append(function(*(a[rows] for a in arrays)))
    values = [v for v in values if not math.isnan(v)]
    return float(np.mean(values)) if values else math.nan


def auc_within(groups, y, score) -> float:
    """The mean of each patient's own AUC, over the patients with both answers; NaN if none has."""
    return within(groups, auc, np.asarray(y, dtype=bool), score, min_classes=2)


# ---- every score of one label -------------------------------------------------------------------

def label_scores(label: str, y, probs, participants, no_dbs: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Every score of one label, on the hours given (the caller leaves out hours that shouldn't count).

    Args:
        label: Such as state_now.
        y: (n,) the true class index per hour; NaN hours are left out.
        probs: (n, classes) the predicted probabilities.
        participants: (n,) whose hour each is.
        no_dbs: (n,) True for the hours of patients without DBS, for off_auc_no_dbs; None to skip it.

    Returns:
        {metric: value}, NaN where a score can't be computed, plus "hours", how many were scored.
    """
    target_name, _ = parse_label(label)
    target = get_target(target_name)
    k = len(target.classes)
    y = np.asarray(y, dtype=float)
    probs = np.asarray(probs, dtype=float)
    participants = np.asarray(participants)
    if probs.shape != (len(y), k):
        raise ConfigError(f"{label}: probabilities are {probs.shape}, need ({len(y)}, {k})")
    if len(participants) != len(y) or (no_dbs is not None and len(no_dbs) != len(y)):
        raise ConfigError(f"{label}: y, participants and no_dbs need one value per hour")
    known = ~np.isnan(y)
    classes_seen = y[known]
    if not (np.all(classes_seen == np.round(classes_seen)) and np.all((classes_seen >= 0) & (classes_seen < k))):
        raise ConfigError(f"{label}: answers must be class indices 0 to {k - 1}")
    if not np.isfinite(probs[known]).all():
        raise ConfigError(f"{label}: probabilities must be finite")
    y, probs, participants = y[known].astype(int), probs[known], participants[known]
    predicted = probs.argmax(axis=1) if len(probs) else np.zeros(0, int)
    out = {"hours": float(len(y))}

    for question, classes in QUESTIONS[target_name].items():
        yes = np.isin(y, classes)
        score = probs[:, list(classes)].sum(axis=1)
        out[f"{question}_auc"] = auc(yes, score)
        out[f"{question}_auc_within"] = auc_within(participants, yes, score)
        if question == "off" and no_dbs is not None:
            keep = np.asarray(no_dbs, dtype=bool)[known]
            out["off_auc_no_dbs"] = auc(yes[keep], score[keep])

    out["balanced_accuracy"] = balanced_accuracy(y, predicted, k)
    out["balanced_accuracy_within"] = within(participants, lambda a, b: balanced_accuracy(a, b, k), y, predicted,
                                             min_classes=2)
    out["macro_f1"] = macro_f1(y, predicted, k)
    out["macro_f1_within"] = within(participants, lambda a, b: macro_f1(a, b, k), y, predicted, min_classes=2)
    m = confusion(y, predicted, k)
    for i, name in enumerate(target.classes):
        answered, guessed = m[i].sum(), m[:, i].sum()
        out[f"recall_{name}"] = float(m[i, i] / answered) if answered else math.nan
        out[f"precision_{name}"] = float(m[i, i] / guessed) if guessed else math.nan

    if all(s is not None for s in target.scores):
        values = np.asarray(target.scores, dtype=float)
        true_score, expected = values[y], probs @ values
        out["score_spearman"] = spearman(true_score, expected)
        out["score_spearman_within"] = within(participants, spearman, true_score, expected, min_classes=2)
        out["score_mae"] = mae(true_score, expected)
        out["score_mae_within"] = within(participants, mae, true_score, expected)
    return out


def mean_scores(per_part: Mapping[str, Mapping[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """{label: {metric: mean}} over parts (folds or seeds), ignoring NaN; NaN where every part is NaN."""
    labels = {label for part in per_part.values() for label in part}
    out = {}
    for label in sorted(labels):
        metrics = {m for part in per_part.values() for m in part.get(label, {})}
        out[label] = {}
        for m in sorted(metrics):
            values = [part[label][m] for part in per_part.values() if label in part and m in part[label]]
            finite = [v for v in values if not math.isnan(v)]
            out[label][m] = float(np.mean(finite)) if finite else math.nan
    return out


def _pair(a, b, name):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ConfigError(f"{name} needs two vectors of the same length, not {a.shape} and {b.shape}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ConfigError(f"{name} needs finite values (a NaN would be ranked as if it were a number)")
    return a, b


def _average_ranks(x):
    """1-based ranks, ties sharing their average rank (no Python loop over the ties)."""
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)                     # each distinct value's last rank
    return ((ends - counts + 1 + ends) / 2.0)[inverse]


__all__ = ["LOWER_IS_BETTER", "QUESTIONS", "auc", "auc_within", "balanced_accuracy", "confusion", "higher_is_better",
           "label_scores", "macro_f1", "mae", "mean_scores", "spearman", "within"]
