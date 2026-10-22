"""Bias detection through data slicing, with a simple mitigation step.

Two levels of analysis for each slice column (sex, age group, ...):
    1. Data level: rows, participants and label balance per group.
    2. Model level: a baseline logistic regression predicting "off" vs
       "not off", trained on some participants and tested on others.
       Recall, precision and false positive rate are compared per group
       (with Fairlearn's MetricFrame when it is installed).

If the recall gap between groups is larger than bias.max_recall_gap,
the model is retrained with sample weights that give each group equal
total weight, and the before/after results are reported.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from scripts.anomalies import feature_columns
from scripts.utils.file_io import write_json
from scripts.utils.logging_setup import get_logger

logger = get_logger("bias")

try:
    from fairlearn.metrics import MetricFrame, false_positive_rate
    from sklearn.metrics import precision_score, recall_score
    HAS_FAIRLEARN = True
except ImportError:  # The pipeline still works without Fairlearn.
    HAS_FAIRLEARN = False


def available_slices(df, cfg):
    cols = [c for c in cfg["bias"]["slice_columns"] if c in df.columns]
    skipped = set(cfg["bias"]["slice_columns"]) - set(cols)
    if skipped:
        logger.warning("Slice columns not in data (check demographics column_map): %s", sorted(skipped))
    return cols


def data_level_slices(df, slice_cols):
    """Label balance and coverage per group."""
    out = {}
    for col in slice_cols:
        groups = df[col].fillna("missing").astype(str)
        table = pd.crosstab(groups, df["kinesia_class"], normalize="index").round(3)
        summary = df.assign(_g=groups).groupby("_g").agg(
            rows=("participant_id", "size"), participants=("participant_id", "nunique"))
        out[col] = summary.join(table).reset_index().rename(columns={"_g": col}).to_dict("records")
    return out


def _manual_metrics(y_true, y_pred, groups):
    rows = []
    for g in sorted(groups.unique()):
        m = groups == g
        t, p = y_true[m], y_pred[m]
        tp, fn = int(((t == 1) & (p == 1)).sum()), int(((t == 1) & (p == 0)).sum())
        fp, tn = int(((t == 0) & (p == 1)).sum()), int(((t == 0) & (p == 0)).sum())
        rows.append({"group": g, "n": int(m.sum()), "positives": tp + fn,
                     "recall": tp / (tp + fn) if tp + fn else np.nan,
                     "precision": tp / (tp + fp) if tp + fp else np.nan,
                     "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan})
    return pd.DataFrame(rows)


def slice_metrics(y_true, y_pred, groups):
    """Per-group recall, precision and false positive rate."""
    base = _manual_metrics(y_true, y_pred, groups)
    if HAS_FAIRLEARN:
        frame = MetricFrame(
            metrics={"recall": lambda t, p: recall_score(t, p, zero_division=np.nan),
                     "precision": lambda t, p: precision_score(t, p, zero_division=np.nan),
                     "false_positive_rate": false_positive_rate},
            y_true=y_true, y_pred=y_pred, sensitive_features=groups)
        by_group = frame.by_group.reset_index()
        by_group = by_group.rename(columns={by_group.columns[0]: "group"})
        base = base[["group", "n", "positives"]].merge(by_group, on="group", how="left")
    return base


def _fit_predict(X_train, y_train, X_test, weights=None, seed=42):
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
    model.fit(X_train, y_train, logisticregression__sample_weight=weights)
    return model.predict(X_test)


def _group_weights(groups):
    """Weights that give every group the same total weight."""
    counts = groups.value_counts()
    return groups.map(lambda g: len(groups) / (len(counts) * counts[g])).to_numpy()


def model_level_slices(df, slice_cols, cfg):
    """Train a baseline model on some participants, compare slices on the rest."""
    b = cfg["bias"]
    n_participants = df["participant_id"].nunique()
    needed = max(2, b["min_participants_for_model"])  # train and test need different people
    if n_participants < needed:
        return {"skipped": f"only {n_participants} participants (need {needed})"}

    y = (df["kinesia_class"] == b["positive_class"]).astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return {"skipped": f"no variation in '{b['positive_class']}' labels"}
    X = df[feature_columns(df)].to_numpy(dtype=float)

    splitter = GroupShuffleSplit(n_splits=1, test_size=b["test_fraction"], random_state=cfg["data"]["random_seed"])
    train_idx, test_idx = next(splitter.split(X, y, groups=df["participant_id"]))
    if len(np.unique(y[train_idx])) < 2:
        return {"skipped": "training split has only one class"}

    preds = _fit_predict(X[train_idx], y[train_idx], X[test_idx], seed=cfg["data"]["random_seed"])
    test = df.iloc[test_idx]
    results = {"test_participants": sorted(test["participant_id"].unique()),
               "overall_recall": _safe_recall(y[test_idx], preds), "slices": {}}

    for col in slice_cols:
        groups = test[col].fillna("missing").astype(str).reset_index(drop=True)
        metrics = slice_metrics(y[test_idx], preds, groups)
        valid = metrics[metrics["n"] >= b["min_slice_rows"]]
        gap = float(valid["recall"].max() - valid["recall"].min()) if valid["recall"].notna().sum() > 1 else np.nan
        entry = {"metrics": metrics.round(3).to_dict("records"), "recall_gap": gap,
                 "biased": bool(gap > b["max_recall_gap"]) if not np.isnan(gap) else False}

        if entry["biased"]:
            train_groups = df.iloc[train_idx][col].fillna("missing").astype(str).reset_index(drop=True)
            new_preds = _fit_predict(X[train_idx], y[train_idx], X[test_idx],
                                     weights=_group_weights(train_groups), seed=cfg["data"]["random_seed"])
            after = slice_metrics(y[test_idx], new_preds, groups)
            valid_after = after[after["n"] >= b["min_slice_rows"]]
            entry["mitigation"] = {
                "method": "reweighting (equal total weight per group)",
                "metrics_after": after.round(3).to_dict("records"),
                "recall_gap_after": float(valid_after["recall"].max() - valid_after["recall"].min())
                if valid_after["recall"].notna().sum() > 1 else np.nan,
                "overall_recall_after": _safe_recall(y[test_idx], new_preds),
            }
        results["slices"][col] = entry
    return results


def _safe_recall(y_true, y_pred):
    positives = (y_true == 1).sum()
    return float(((y_true == 1) & (y_pred == 1)).sum() / positives) if positives else np.nan


def run_bias_analysis(df, cfg):
    """Run both levels of slicing and write JSON and Markdown reports."""
    slice_cols = available_slices(df, cfg)
    report = {"slice_columns": slice_cols, "fairlearn_used": HAS_FAIRLEARN,
              "data_level": data_level_slices(df, slice_cols),
              "model_level": model_level_slices(df, slice_cols, cfg)}
    reports = cfg["resolved_paths"]["reports_dir"]
    write_json(report, reports / "bias_report.json")
    (reports / "bias_report.md").write_text(to_markdown(report), encoding="utf-8")
    logger.info("Bias report written for slices %s", slice_cols)
    return report


def _md_table(records):
    if not records:
        return "_no data_\n"
    cols = list(records[0].keys())
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in records:
        lines.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def to_markdown(report):
    out = ["# Bias report\n", f"Slices: {', '.join(report['slice_columns']) or 'none'}\n",
           "## Data level\n"]
    for col, records in report["data_level"].items():
        out += [f"### {col}\n", _md_table(records)]
    out.append("## Model level (baseline: off vs not off)\n")
    model = report["model_level"]
    if "skipped" in model:
        out.append(f"Skipped: {model['skipped']}\n")
        return "\n".join(out)
    out.append(f"Overall recall: {model['overall_recall']:.3f}\n")
    for col, entry in model["slices"].items():
        out += [f"### {col}\n", _md_table(entry["metrics"]),
                f"Recall gap: {entry['recall_gap']:.3f} -> "
                f"{'**bias flagged**' if entry['biased'] else 'within limit'}\n"]
        if "mitigation" in entry:
            m = entry["mitigation"]
            out += [f"Mitigation: {m['method']}\n", _md_table(m["metrics_after"]),
                    f"Recall gap after: {m['recall_gap_after']:.3f}; "
                    f"overall recall {model['overall_recall']:.3f} -> {m['overall_recall_after']:.3f}\n"]
    return "\n".join(out)
