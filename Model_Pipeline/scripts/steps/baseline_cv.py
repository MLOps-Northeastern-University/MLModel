"""The hourly baseline's repeated cross-validation and its report (`run_model_pipeline --baseline`).

Owner: P1.

The development patients (the split file's, test patients never touched) are dealt into
baseline.cv_folds folds by patient, with folds.make_folds (DBS and dyskinetic patients spread evenly),
and every fold is predicted by a model trained on the others. That is repeated baseline.repeats
times, each with a new shuffle (seed + repeat), so the score doesn't hang on one lucky or unlucky
split. The split file itself is never changed.

Every metric is computed once per repeat, on that repeat's out-of-fold hours (labelled and worn),
then summarised over the repeats as mean, sd, min and max. Best-of-repeats is never reported: it's
the luckiest shuffle, not the model.

Per question (OFF, ON, dyskinetic, tremor, asleep; now and next hour), from P(question):
    roc_auc, roc_auc_within    rank quality, pooled and per patient (chance 0.5)
    pr_auc                     average precision; chance is the prevalence, reported next to it
    pr_lift                    pr_auc / prevalence: how many times better than chance
    brier                      mean squared error of the probability (lower is better)
    sensitivity, specificity, precision, f1   at probability >= threshold
Per label, the team's scores (core/metrics.py): balanced_accuracy, macro_f1, score_spearman(_within)...

Then the final model is fitted on all development patients and saved, and its standardised
coefficients show what drives each answer.

Writes <artifacts_dir>/baseline/:
    repeats.csv        one row per repeat and metric
    summary.csv        per metric: mean, sd, min, max
    coefficients.csv   the final model's coefficients per label and class
    report.md          all of it, readable
    curves.png         ROC and precision-recall curves of OFF and tremor (first repeat)
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.core.errors import ConfigError
from scripts.core.metrics import QUESTIONS, auc, auc_within
from scripts.core.targets import PARTICIPANT, get_target, parse_label
from scripts.data.baseline_loader import FEATURE_COLUMNS
from scripts.data.folds import Split, make_folds, patient_facts
from scripts.experiments import cross_validate, fresh, load_development
from scripts.models.registry import build
from scripts.utils.file_io import write_json
from scripts.utils.logging_setup import get_logger

logger = get_logger(__name__)

DEFAULTS = {"cv_folds": 5, "repeats": 5, "seed": 42, "threshold": 0.5}
HEADLINE = ["state_now/off", "state_now/dyskinetic", "tremor_now/any_tremor", "sleep_now/asleep",
            "state_next_hour/off", "tremor_next_hour/any_tremor"]


def settings(cfg):
    section = cfg.get("baseline") or {}
    if not section.get("candidate"):
        raise ConfigError("--baseline needs baseline.candidate in the config")
    out = {**DEFAULTS, **{k: section[k] for k in DEFAULTS if k in section}}
    for key in ("cv_folds", "repeats"):
        if isinstance(out[key], bool) or not isinstance(out[key], int) or out[key] < (2 if key == "cv_folds" else 1):
            raise ConfigError(f"baseline.{key} must be a whole number of {2 if key == 'cv_folds' else 1} or more")
    if not 0 < out["threshold"] < 1:
        raise ConfigError("baseline.threshold must be between 0 and 1")
    return out


# ---- metrics ---------------------------------------------------------------------------------------

def question_metrics(y, p, participants, threshold):
    """Every binary score of one question: y is yes/no per hour, p the model's P(yes)."""
    from sklearn.metrics import average_precision_score
    y = np.asarray(y, dtype=bool)
    p = np.asarray(p, dtype=float)
    prevalence = float(y.mean()) if len(y) else math.nan
    both = 0 < y.sum() < len(y)
    flagged = p >= threshold
    tp, fp = int((flagged & y).sum()), int((flagged & ~y).sum())
    fn, tn = int((~flagged & y).sum()), int((~flagged & ~y).sum())
    sens = tp / (tp + fn) if tp + fn else math.nan
    prec = tp / (tp + fp) if tp + fp else math.nan
    pr_auc = float(average_precision_score(y, p)) if both else math.nan
    return {
        "hours": int(len(y)), "prevalence": prevalence,
        "roc_auc": auc(y, p), "roc_auc_within": auc_within(participants, y, p),
        "pr_auc": pr_auc, "pr_lift": pr_auc / prevalence if both else math.nan,
        "brier": float(np.mean((p - y) ** 2)) if len(y) else math.nan,
        "sensitivity": sens, "specificity": tn / (tn + fp) if tn + fp else math.nan, "precision": prec,
        "f1": 2 * prec * sens / (prec + sens) if prec and sens and not math.isnan(prec + sens) else math.nan,
    }


def frame_metrics(frames, threshold):
    """{"<label>/<question>/<metric>": value} on the scored hours (labelled and worn) of one repeat."""
    out = {}
    for label, frame in frames.items():
        target = parse_label(label)[0]
        classes = list(get_target(target).classes)
        f = frame[frame["worn"].to_numpy(bool) & frame["y_true"].notna()]
        y = f["y_true"].to_numpy(int)
        probs = f[classes].to_numpy(float)
        for question, members in QUESTIONS[target].items():
            scores = question_metrics(np.isin(y, members), probs[:, list(members)].sum(axis=1),
                                      f[PARTICIPANT].to_numpy(), threshold)
            out.update({f"{label}/{question}/{k}": v for k, v in scores.items()})
    return out


def tracker_name(metric, suffix=""):
    """"state_now/off/roc_auc" -> "cv/state_now/off_roc_auc" (the tracker's {split}/{label}/{metric})."""
    label, *rest = metric.split("/")
    return f"cv/{label}/{'_'.join(rest)}{suffix}"


def summarise(repeats):
    """repeats: one row per repeat and metric -> per metric mean, sd, min, max."""
    g = repeats.groupby("metric")["value"]
    return pd.DataFrame({"mean": g.mean(), "sd": g.std(), "min": g.min(), "max": g.max(),
                         "repeats": g.count()}).reset_index()


# ---- the run ---------------------------------------------------------------------------------------

def run(cfg, tracker, data=None, split=None):
    """Repeated CV of the baseline, the final model, and the report. Returns the summary table."""
    from scripts.data.folds import read_split
    s = settings(cfg)
    name, template = build(cfg["baseline"]["candidate"], None, cfg["models"].get("targets"),
                           cfg["models"].get("horizons"), where="baseline.candidate")
    split = split or read_split(cfg)
    if data is None:
        data, _ = load_development(cfg, split, [(name, template)])
    dev = data.subset(split.development)
    facts = patient_facts(dev)
    out = Path(cfg["resolved_paths"]["artifacts_dir"]) / "baseline"
    out.mkdir(parents=True, exist_ok=True)

    rows, first_frames = [], None
    with tracker.run("baseline", tags={"owner": "P1", "split_hash": split.hash}):
        tracker.log_params({"baseline": s, "candidate": cfg["baseline"]["candidate"]})
        for r in range(s["repeats"]):
            folds = make_folds(facts, s["cv_folds"], s["seed"] + r)
            repeat_split = Split(split.test, folds, s["seed"] + r)
            with tracker.run(f"repeat-{r + 1}"):
                result = cross_validate(name, template, dev, repeat_split, [s["seed"]], tracker)
                metrics = frame_metrics(result.frames, s["threshold"])
                metrics.update({f"{label}/{m}": v for label, ms in result.cv.items() for m, v in ms.items()})
                tracker.log_metrics({tracker_name(k): v for k, v in metrics.items()
                                     if isinstance(v, (int, float)) and math.isfinite(v)})
            rows += [{"repeat": r + 1, "metric": k, "value": v} for k, v in metrics.items()]
            first_frames = first_frames or result.frames
            logger.info("baseline repeat %d/%d: OFF ROC AUC %.3f, PR AUC %.3f", r + 1, s["repeats"],
                        metrics.get("state_now/off/roc_auc", math.nan), metrics.get("state_now/off/pr_auc", math.nan),
                        extra={"event": "baseline_repeat", "repeat": r + 1})
        repeats = pd.DataFrame(rows)
        summary = summarise(repeats)
        for stat in ("mean", "sd"):
            tracker.log_metrics({tracker_name(m, f"_{stat}"): v for m, v in zip(summary["metric"], summary[stat])
                                 if math.isfinite(v)})

        final = fresh(template).fit(dev, seed=s["seed"])
        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d-%H%M%S")
        model_dir = final.save(Path(cfg["resolved_paths"]["artifacts_dir"]) / "models" / f"{name}-{stamp}")
        coefficients = coefficient_table(final)

    repeats.to_csv(out / "repeats.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    coefficients.to_csv(out / "coefficients.csv", index=False)
    write_json({"settings": s, "candidate": cfg["baseline"]["candidate"], "model_dir": str(model_dir),
                "development_patients": len(dev.participants), "test_patients_used": 0,
                "summary": summary.to_dict(orient="records")}, out / "summary.json")
    plot_curves(first_frames, out / "curves.png")
    (out / "report.md").write_text(report(summary, coefficients, s, name, dev, model_dir), encoding="utf-8")
    logger.info("baseline report written to %s", out, extra={"event": "baseline_report", "path": str(out)})
    return summary


def coefficient_table(model):
    """The final model's coefficients on the standardised features: one row per label, class and feature."""
    rows = []
    for label, pipeline in model._classifiers.items():
        if pipeline is None:
            continue
        lr = pipeline[-1]
        target = parse_label(label)[0]
        names = ["off", "on", "dyskinetic"] if target == "state" else list(get_target(target).classes)
        coef = lr.coef_
        if coef.shape[0] == 1:                    # binary: one row, for the second class
            classes = [names[int(lr.classes_[1])]]
        else:
            classes = [names[int(c)] for c in lr.classes_]
        for cls, row in zip(classes, coef):
            rows += [{"label": label, "class": cls, "feature": f, "coefficient": float(v)}
                     for f, v in zip(FEATURE_COLUMNS, row)]
    return pd.DataFrame(rows)


# ---- the report ------------------------------------------------------------------------------------

def _cell(summary, metric, digits=3):
    row = summary[summary["metric"] == metric]
    if row.empty or not math.isfinite(row["mean"].iloc[0]):
        return "–"
    m, sd, lo, hi = row[["mean", "sd", "min", "max"]].iloc[0]
    sd = 0.0 if not math.isfinite(sd) else sd
    return f"{m:.{digits}f} ± {sd:.{digits}f} [{lo:.{digits}f}–{hi:.{digits}f}]"


def _hours(summary, question):
    """Scored hours: the same every repeat (every development hour is scored once per repeat)."""
    row = summary[summary["metric"] == f"{question}/hours"]
    return "–" if row.empty else f"{row['mean'].iloc[0]:.0f}"


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def report(summary, coefficients, s, name, dev, model_dir):
    questions = sorted({m.rsplit("/", 1)[0] for m in summary["metric"] if m.count("/") == 2},
                       key=lambda q: (q not in HEADLINE, HEADLINE.index(q) if q in HEADLINE else 0, q))
    ranking = [[q, _hours(summary, q), _cell(summary, f"{q}/prevalence"), _cell(summary, f"{q}/roc_auc"),
                _cell(summary, f"{q}/roc_auc_within"), _cell(summary, f"{q}/pr_auc"), _cell(summary, f"{q}/pr_lift", 2)]
               for q in questions]
    calibration = [[q, _cell(summary, f"{q}/brier"), _cell(summary, f"{q}/sensitivity"),
                    _cell(summary, f"{q}/specificity"), _cell(summary, f"{q}/precision"), _cell(summary, f"{q}/f1")]
                   for q in questions]
    labels = sorted({m.split("/")[0] for m in summary["metric"]})
    team = [[label] + [_cell(summary, f"{label}/{m}") for m in
                       ("balanced_accuracy", "macro_f1", "score_spearman", "score_spearman_within", "score_mae")]
            for label in labels]
    top = []
    for (label, cls), g in coefficients.groupby(["label", "class"]):
        if label.endswith("_now") and (label, cls) != ("state_now", "on"):
            best = g.reindex(g["coefficient"].abs().sort_values(ascending=False).index).head(4)
            top.append([label, cls, ", ".join(f"{f} {c:+.2f}" for f, c in zip(best["feature"], best["coefficient"]))])
    return f"""# Hourly baseline: {name}

{s['repeats']} repeats of {s['cv_folds']}-fold cross-validation by patient, over {len(dev.participants)}
development patients (about {len(dev.participants) * (s['cv_folds'] - 1) // s['cv_folds']} train and
{len(dev.participants) // s['cv_folds']} validation per fold). The test patients were not used.

Each cell: mean ± sd [min–max] over the repeats. Best-of-repeats isn't reported.
Only labelled hours with a worn watch are scored.

## Ranking: does it put the right hours on top?

ROC AUC: chance 0.5. PR AUC: chance is the prevalence; lift = PR AUC / prevalence.
"within" is each patient's own AUC, averaged: following one person through the day.

{_table(["question", "hours", "prevalence", "ROC AUC", "ROC AUC within", "PR AUC", "PR lift"], ranking)}

## At probability ≥ {s['threshold']}

Brier: mean squared error of the probability (lower is better). These depend on the threshold, and
balanced class weights shift the probabilities towards the rare classes, so read them as one
operating point; the AUCs above don't depend on it.

{_table(["question", "Brier", "sensitivity", "specificity", "precision", "F1"], calibration)}

## The team's scores (core/metrics.py)

{_table(["label", "balanced accuracy", "macro F1", "Spearman", "Spearman within", "MAE"], team)}

## What drives it (final model, standardised coefficients)

Fitted on all {len(dev.participants)} development patients, saved to `{model_dir}`. A positive
coefficient raises that class's odds. Full table: coefficients.csv.

{_table(["label", "class", "largest 4"], top)}

Curves (first repeat): curves.png.
"""


def plot_curves(frames, path):
    """ROC and precision-recall curves of OFF and any tremor (now), pooled over the patients."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for label, question in (("state_now", "off"), ("tremor_now", "any_tremor"), ("state_now", "dyskinetic")):
        if label not in frames:
            continue
        target = parse_label(label)[0]
        members = list(QUESTIONS[target][question])
        classes = list(get_target(target).classes)
        f = frames[label][frames[label]["worn"].to_numpy(bool) & frames[label]["y_true"].notna()]
        y = np.isin(f["y_true"].to_numpy(int), members)
        if not 0 < y.sum() < len(y):
            continue
        p = f[[classes[i] for i in members]].to_numpy(float).sum(axis=1)
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        line, = axes[0].plot(fpr, tpr, label=f"{question} (AUC {auc(y, p):.2f})")
        axes[1].plot(rec, prec, color=line.get_color(), label=question)
        axes[1].axhline(y.mean(), color=line.get_color(), linestyle=":", linewidth=1)
    axes[0].plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC")
    axes[1].set(xlabel="recall", ylabel="precision", title="Precision-recall (dotted: chance)")
    for ax in axes:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
