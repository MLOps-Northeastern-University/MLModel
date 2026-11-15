"""The charts, as matplotlib figures: drawing only, nothing saved or logged (writer.py does that).

Owner: P2.

    scorecard_figure(scores, title)          one run's scores: a panel per label, each question's AUC,
                                             balanced accuracy, macro F1 and the diary-score correlation,
                                             pooled and within side by side, with chance marked
    confusion_figure(label, frame, title)    one label's confusion matrix: each row a true class, each
                                             cell the share of that class's hours predicted as the column
    comparison_figure(results, metrics)      the candidates side by side, a panel per metric, with each
                                             candidate's spread across folds as an error bar

The inputs are what experiments.py already makes: {label: {metric: value}} scores, the prediction
rows (one per hour, y_true and a column per class) and tracker metric names. Each function returns a
Figure that the caller must close (writer.py does).

matplotlib draws with Agg, which needs no display, so this runs on a server or in CI.
"""

import math
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from scripts.core.metrics import QUESTIONS, confusion, higher_is_better
from scripts.core.targets import get_target, parse_label

# The scores a scorecard shows, in order, with their chance level (None: no single chance level).
# score_mae is left out: it's in score points, not on the 0-1 scale of the others (it's in MLflow).
CLASS_SCORES = ("balanced_accuracy", "macro_f1", "score_spearman")
POOLED, WITHIN = "#2f6f4f", "#9cc5a1"      # the two versions of each score
CHANCE = "#333333"
DPI = 150


def pyplot():
    """matplotlib.pyplot on the Agg backend (no display needed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def chance_level(label: str, metric: str) -> Optional[float]:
    """What guessing scores: 0.5 for an AUC, 1/classes for balanced accuracy, 0 for a correlation.

    The same for a score's within version (off_auc_within) as for its pooled one.
    """
    metric = metric[:-len("_within")] if metric.endswith("_within") else metric
    if metric.endswith("_auc") or metric.endswith("_auc_no_dbs"):
        return 0.5
    if metric == "balanced_accuracy":
        return 1.0 / len(get_target(parse_label(label)[0]).classes)
    if metric == "score_spearman":
        return 0.0
    return None


def scorecard_rows(label: str, scores: Mapping[str, float]):
    """[(metric, pooled, within)] for one label, in scorecard order, NaN where a score is missing."""
    target = parse_label(label)[0]
    metrics = [f"{q}_auc" for q in QUESTIONS.get(target, {})] + list(CLASS_SCORES)
    out = []
    for metric in metrics:
        pooled, within = scores.get(metric, math.nan), scores.get(f"{metric}_within", math.nan)
        if _finite(pooled) or _finite(within):
            out.append((metric, float(pooled), float(within)))
    return out


def scorecard_figure(scores: Mapping[str, Mapping[str, float]], title: str):
    """One run's scores, a panel per label (see the top of this file)."""
    plt = pyplot()
    labels = [label for label in scores if scorecard_rows(label, scores[label])]
    if not labels:
        raise ValueError("no scores to draw")
    columns = min(2, len(labels))
    grid_rows = math.ceil(len(labels) / columns)
    figure, axes = plt.subplots(grid_rows, columns, figsize=(6.5 * columns, 2.6 * grid_rows), squeeze=False)
    for ax, label in zip(axes.flat, labels):
        rows = scorecard_rows(label, scores[label])
        y = np.arange(len(rows))
        pooled = np.array([r[1] for r in rows])
        within = np.array([r[2] for r in rows])
        ax.barh(y - 0.2, np.nan_to_num(pooled), height=0.38, color=POOLED, label="pooled")
        ax.barh(y + 0.2, np.nan_to_num(within), height=0.38, color=WITHIN, label="within patients")
        for i, (metric, p, w) in enumerate(rows):
            for offset, value in ((-0.2, p), (0.2, w)):
                text = f"{value:.2f}" if _finite(value) else "n/a"
                ax.text(max(value, 0) + 0.01 if _finite(value) else 0.01, i + offset, text, va="center", fontsize=7)
            level = chance_level(label, metric)
            if level is not None:
                ax.plot([level, level], [i - 0.45, i + 0.45], color=CHANCE, linewidth=1.2, linestyle=":")
        ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
        ax.invert_yaxis()
        low = min(0.0, np.nanmin(np.concatenate([pooled, within])) - 0.05)
        ax.set_xlim(low, 1.12)
        ax.set_title(f"{label}  ({int(scores[label].get('hours', 0))} hours)", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
    for ax in list(axes.flat)[len(labels):]:
        ax.set_visible(False)
    handles, names = axes.flat[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=CHANCE, linestyle=":"))
    names.append("chance")
    figure.legend(handles, names, loc="lower center", ncol=3, fontsize=8, frameon=False)
    figure.suptitle(title, fontsize=11)
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    return figure


def confusion_counts(label: str, frame: pd.DataFrame):
    """(counts, class names, seeds): the confusion matrix of the scored hours (labelled and worn).

    Each hour is predicted as its most likely class. Rows are counted once per seed, so with several
    seeds the counts are summed over them; seeds says by how much.
    """
    target = get_target(parse_label(label)[0])
    classes = list(target.classes)
    scored = frame[frame["worn"].to_numpy(bool) & frame["y_true"].notna().to_numpy()]
    y = scored["y_true"].to_numpy(float).astype(int)
    predicted = scored[classes].to_numpy(float).argmax(axis=1)
    seeds = int(scored["seed"].nunique()) if "seed" in scored and len(scored) else 1
    return confusion(y, predicted, len(classes)), class_names(label), max(seeds, 1)


def class_names(label: str) -> list:
    """Short class names for a chart: the diary score where there is one (-3 .. +3), else the name."""
    target = get_target(parse_label(label)[0])
    return [f"{s:+d}" if s and target.name == "state" else (str(s) if s is not None else c)
            for c, s in zip(target.classes, target.scores)]


def confusion_figure(label: str, frame: pd.DataFrame, title: str):
    """One label's confusion matrix, as shares of each true class (see the top of this file)."""
    plt = pyplot()
    counts, names, seeds = confusion_counts(label, frame)
    totals = counts.sum(axis=1, keepdims=True)
    shares = np.divide(counts, totals, out=np.zeros(counts.shape, dtype=float), where=totals > 0)
    k = len(names)
    figure, ax = plt.subplots(figsize=(1.0 + 0.7 * k, 0.8 + 0.6 * k))
    ax.imshow(shares, cmap="Greens", vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            if counts[i, j]:
                ax.text(j, i, f"{shares[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if shares[i, j] > 0.6 else "black")
    per_seed = (totals[:, 0] / seeds).round().astype(int)
    ax.set_xticks(range(k), names, fontsize=7)
    ax.set_yticks(range(k), [f"{n}  (n={c})" for n, c in zip(names, per_seed)], fontsize=7)
    ax.set_xlabel("predicted (most likely class)", fontsize=8)
    ax.set_ylabel("diary answer", fontsize=8)
    seeds_note = f", hours counted per seed over {seeds} seeds" if seeds > 1 else ""
    ax.set_title(f"{title}\n{label}: share of each answer{seeds_note}", fontsize=8)
    figure.tight_layout()
    return figure


def comparison_figure(results: Mapping[str, Mapping[str, float]], metrics: Sequence[str], title: str):
    """The candidates side by side (see the top of this file).

    Args:
        results: {candidate: tracker metrics}, such as CVResult.metrics(): cv/<label>/<metric> and,
            for the error bars, the same name with _fold_sd.
        metrics: Tracker names to show, one panel each. Missing ones are skipped.
    """
    plt = pyplot()
    shown = [m for m in metrics if any(_finite(r.get(m)) for r in results.values())]
    if not shown:
        raise ValueError(f"none of {list(metrics)} were scored")
    names = list(results)
    columns = min(3, len(shown))
    grid_rows = math.ceil(len(shown) / columns)
    figure, axes = plt.subplots(grid_rows, columns, figsize=(4.2 * columns, 0.9 + 0.45 * len(names) * grid_rows + 0.6),
                                squeeze=False)
    for ax, metric in zip(axes.flat, shown):
        values = np.array([results[n].get(metric, math.nan) for n in names], dtype=float)
        spread = np.array([results[n].get(f"{metric}_fold_sd", 0.0) for n in names], dtype=float)
        best = _best(values, higher_is_better(metric))
        colours = [POOLED if i == best else WITHIN for i in range(len(names))]
        y = np.arange(len(names))
        ax.barh(y, np.nan_to_num(values), xerr=np.nan_to_num(spread), color=colours, capsize=3,
                error_kw={"elinewidth": 0.8})
        for i, v in enumerate(values):
            ax.text(0.01, i, f"{v:.3f}" if _finite(v) else "n/a", va="center", fontsize=7,
                    color="white" if i == best else "black")
        label, short = _split_metric(metric)
        level = chance_level(label, short) if label else None
        if level is not None:
            ax.axvline(level, color=CHANCE, linestyle=":", linewidth=1)
        if higher_is_better(metric):                    # a 0-1 score: the whole scale, so bars compare by eye
            low = np.nanmin(np.concatenate([values - spread, [0.0]]))
            ax.set_xlim(min(0.0, low - 0.05), 1.05)
        ax.set_yticks(y, names, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(metric + ("" if higher_is_better(metric) else "  (lower is better)"), fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
    for ax in list(axes.flat)[len(shown):]:
        ax.set_visible(False)
    figure.suptitle(f"{title}\nbar: mean over seeds; line: SD across folds; dark: best; dotted: chance", fontsize=9)
    figure.tight_layout()
    return figure


def _split_metric(name: str):
    """cv/state_now/off_auc -> ("state_now", "off_auc"); (None, name) if it isn't split/label/metric."""
    parts = name.split("/")
    if len(parts) == 3:
        try:
            parse_label(parts[1])
            return parts[1], parts[2]
        except Exception:      # not a label: no chance level to draw
            pass
    return None, name


def _best(values: np.ndarray, higher: bool) -> Optional[int]:
    finite = np.isfinite(values)
    if not finite.any():
        return None
    masked = np.where(finite, values, -np.inf if higher else np.inf)
    return int(np.argmax(masked) if higher else np.argmin(masked))


def _finite(value) -> bool:
    return isinstance(value, (int, float, np.floating)) and math.isfinite(value)


__all__ = ["CLASS_SCORES", "DPI", "chance_level", "class_names", "comparison_figure", "confusion_counts",
           "confusion_figure", "pyplot", "scorecard_figure", "scorecard_rows"]
