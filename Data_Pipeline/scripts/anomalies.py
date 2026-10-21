"""Detect data anomalies and decide whether the pipeline should stop.

Checks: missing values, outliers (robust z-score), participants with too
little data, label problems found while parsing diaries, and schema or
expectation failures from validate.py.
"""

import numpy as np
import pandas as pd

from scripts.utils.file_io import write_json
from scripts.utils.logging_setup import get_logger

logger = get_logger("anomalies")

FEATURE_PREFIXES = ("L_move", "R_move", "L_tremor", "R_tremor", "L_still", "R_still", "asym_")


class DataAnomalyError(RuntimeError):
    """Raised when a critical anomaly is found."""


def missing_value_anomalies(df, cfg):
    limit = cfg["anomalies"]["max_missing_fraction"]
    out = []
    for col in feature_columns(df):
        frac = float(df[col].isna().mean())
        if frac > limit:
            out.append({"type": "missing_values", "column": col,
                        "detail": f"{frac:.1%} missing (limit {limit:.0%})"})
    return out


def outlier_anomalies(df, cfg):
    """Flag columns where many values are far from the median (median/MAD z-score)."""
    a = cfg["anomalies"]
    out = []
    for col in feature_columns(df):
        s = df[col].dropna()
        if len(s) < 10:
            continue
        mad = np.median(np.abs(s - s.median()))
        if mad == 0:
            continue
        z = 0.6745 * (s - s.median()) / mad
        frac = float((z.abs() > a["outlier_z_limit"]).mean())
        if frac > a["max_outlier_fraction"]:
            out.append({"type": "outliers", "column": col,
                        "detail": f"{frac:.1%} of values beyond |z| > {a['outlier_z_limit']}"})
    return out


def coverage_anomalies(df, cfg):
    minimum = cfg["anomalies"]["min_hours_per_participant"]
    if df.empty:
        return [{"type": "empty_dataset", "detail": "no rows after cleaning"}]
    counts = df.groupby("participant_id").size()
    return [{"type": "low_coverage", "participant_id": pid,
             "detail": f"only {n} usable hours (minimum {minimum})"}
            for pid, n in counts.items() if n < minimum]


def label_issue_summary(issues):
    """Summarise diary label issues (e.g. text and score disagree)."""
    if not issues:
        return []
    counts = pd.DataFrame(issues).groupby(["field", "issue"]).size()
    return [{"type": "label_issue", "field": field, "issue": issue, "detail": f"{n} rows"}
            for (field, issue), n in counts.items()]


def detect_anomalies(df, cfg, label_issues=None, validation_violations=None):
    """Run every check and return a report with critical and warning lists."""
    found = (missing_value_anomalies(df, cfg) + outlier_anomalies(df, cfg)
             + coverage_anomalies(df, cfg) + label_issue_summary(label_issues or [])
             + list(validation_violations or []))
    critical_types = set(cfg["anomalies"]["critical_types"])
    report = {
        "critical": [a for a in found if a["type"] in critical_types],
        "warnings": [a for a in found if a["type"] not in critical_types],
    }
    write_json(report, cfg["resolved_paths"]["reports_dir"] / "anomalies.json")
    logger.info("Anomalies: %d critical, %d warnings", len(report["critical"]), len(report["warnings"]))
    return report


def format_report(report, limit=15):
    """Readable text version of the report for alerts."""
    lines = []
    for level in ["critical", "warnings"]:
        items = report.get(level, [])
        if items:
            lines.append(f"{level.upper()} ({len(items)}):")
            for a in items[:limit]:
                where = a.get("column") or a.get("participant_id") or a.get("field") or ""
                lines.append(f"  - {a['type']} {where}: {a.get('detail', '')}")
            if len(items) > limit:
                lines.append(f"  ... and {len(items) - limit} more")
    return "\n".join(lines) or "No anomalies found."


def feature_columns(df):
    return [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
