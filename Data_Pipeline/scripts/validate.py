"""Data schema, expectations and statistics.

Three layers of checks run on the processed dataset:
    1. Expectations: fixed rules that must always hold (e.g. labels are
       one of off/on/dyskinetic, fractions are between 0 and 1).
    2. Schema: inferred from the first run and saved as a baseline. Later
       runs are compared with it to catch drift in columns, types, ranges
       and categories.
    3. Statistics: summary stats saved for every run, plus a history file,
       so data quality can be compared over time.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from scripts.utils.file_io import read_json, write_json
from scripts.utils.logging_setup import get_logger

logger = get_logger("validate")

FRACTION_COLUMNS = ["L_still_fraction", "R_still_fraction", "L_valid_fraction", "R_valid_fraction"]
NON_NEGATIVE_COLUMNS = [
    f"{w}_{f}" for w in "LR"
    for f in ["move_mean", "move_std", "move_p95", "tremor_mean", "tremor_p95", "tremor_ratio"]
]


def check_expectations(df, cfg):
    """Run the fixed data rules. Returns a list of failed expectations."""
    failures = []

    def expect(condition, name, detail):
        if not condition:
            failures.append({"type": "expectation_failed", "expectation": name, "detail": detail})

    expect(len(df) > 0, "dataset_not_empty", f"{len(df)} rows")
    for col in ["participant_id", "hour_index", "kinesia_class"]:
        expect(col in df.columns and df[col].notna().all(), f"{col}_not_null",
               f"{df[col].isna().sum() if col in df else 'missing'} nulls")

    allowed = set(cfg["labels"]["kinesia_classes"])
    if "kinesia_class" in df:
        bad = set(df["kinesia_class"].dropna()) - allowed
        expect(not bad, "kinesia_class_in_set", f"unexpected values {sorted(bad)}")
    if "kinesia_score" in df:
        scores = df["kinesia_score"].dropna()
        expect(scores.between(-3, 3).all(), "kinesia_score_in_range", "outside -3..3")
    for col in FRACTION_COLUMNS:
        if col in df:
            expect(df[col].dropna().between(0, 1).all(), f"{col}_between_0_and_1", "outside 0..1")
    for col in NON_NEGATIVE_COLUMNS:
        if col in df:
            expect((df[col].dropna() >= 0).all(), f"{col}_non_negative", "negative values")
    if {"participant_id", "hour_index"}.issubset(df.columns):
        dupes = int(df.duplicated(["participant_id", "hour_index"]).sum())
        expect(dupes == 0, "unique_participant_hour", f"{dupes} duplicates")
    return failures


def infer_schema(df, cfg):
    """Describe each column: type, null rate, numeric range or categories.

    ID columns (validation.id_columns) get no category list, because every
    new participant would otherwise look like a new category.
    """
    max_unique = cfg["validation"]["categorical_max_unique"]
    id_columns = set(cfg["validation"]["id_columns"])
    columns = {}
    for col in df.columns:
        s = df[col]
        info = {"dtype": "numeric" if pd.api.types.is_numeric_dtype(s) else "categorical",
                "null_fraction": float(s.isna().mean())}
        if info["dtype"] == "numeric":
            info["min"] = None if s.dropna().empty else float(s.min())
            info["max"] = None if s.dropna().empty else float(s.max())
        elif col not in id_columns and s.nunique(dropna=True) <= max_unique:
            info["allowed_values"] = sorted(map(str, s.dropna().unique()))
        columns[col] = info
    return {"created_at": datetime.now(timezone.utc).isoformat(), "n_rows": len(df), "columns": columns}


def validate_against_schema(df, schema, cfg):
    """Compare a dataset with a baseline schema. Returns a list of violations."""
    v = cfg["validation"]
    violations = []
    base_cols = schema["columns"]

    for col in base_cols:
        if col not in df.columns:
            violations.append({"type": "schema_violation", "column": col, "detail": "column missing"})
    for col in df.columns:
        if col not in base_cols:
            violations.append({"type": "schema_warning", "column": col, "detail": "new column"})

    for col, info in base_cols.items():
        if col not in df.columns:
            continue
        s = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(s)
        if info["dtype"] == "numeric" and not is_numeric and s.notna().any():
            violations.append({"type": "schema_violation", "column": col,
                               "detail": "expected numeric values"})
            continue
        null_increase = float(s.isna().mean()) - info["null_fraction"]
        if null_increase > v["max_null_fraction_increase"]:
            violations.append({"type": "schema_warning", "column": col,
                               "detail": f"null fraction up by {null_increase:.1%}"})
        if info["dtype"] == "numeric" and info.get("min") is not None and is_numeric and s.notna().any():
            span = (info["max"] - info["min"]) or abs(info["max"]) or 1.0
            slack = span * v["range_tolerance"]
            if s.min() < info["min"] - slack or s.max() > info["max"] + slack:
                violations.append({"type": "schema_warning", "column": col,
                                   "detail": f"range [{s.min():.4g}, {s.max():.4g}] outside "
                                             f"baseline [{info['min']:.4g}, {info['max']:.4g}]"})
        if "allowed_values" in info:
            new = set(map(str, s.dropna().unique())) - set(info["allowed_values"])
            if new:
                violations.append({"type": "schema_warning", "column": col,
                                   "detail": f"new categories {sorted(new)}"})
    return violations


def generate_statistics(df, cfg):
    """Summary statistics for every column plus the label distribution."""
    stats = {"generated_at": datetime.now(timezone.utc).isoformat(),
             "n_rows": len(df),
             "n_participants": int(df["participant_id"].nunique()) if "participant_id" in df else 0,
             "columns": {}}
    for col in df.columns:
        s = df[col]
        entry = {"null_fraction": float(s.isna().mean()), "n_unique": int(s.nunique())}
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            q = s.quantile([0.25, 0.5, 0.75])
            entry.update({"mean": float(s.mean()), "std": float(s.std()) if len(s.dropna()) > 1 else 0.0,
                          "min": float(s.min()), "p25": float(q.iloc[0]), "median": float(q.iloc[1]),
                          "p75": float(q.iloc[2]), "max": float(s.max())})
        else:
            entry["top_values"] = {str(k): int(v) for k, v in s.value_counts().head(10).items()}
        stats["columns"][col] = entry
    if "kinesia_class" in df:
        stats["label_distribution"] = {str(k): int(v) for k, v in df["kinesia_class"].value_counts().items()}
    return stats


def run_validation(df, cfg, run_id="manual"):
    """Run all checks, save the schema, stats and history, return violations."""
    reports = cfg["resolved_paths"]["reports_dir"]
    baseline_path = reports / "schema_baseline.json"

    schema = infer_schema(df, cfg)
    write_json(schema, reports / "schema_latest.json")
    if baseline_path.exists():
        violations = validate_against_schema(df, read_json(baseline_path), cfg)
    else:
        write_json(schema, baseline_path)
        logger.info("No baseline schema found; saved this run as the baseline")
        violations = []
    violations += check_expectations(df, cfg)

    stats = generate_statistics(df, cfg)
    safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in run_id)
    write_json(stats, reports / "stats" / f"stats_{safe_id}.json")
    _append_history(stats, reports / "stats_history.csv", run_id)
    write_json(violations, reports / "validation_report.json")

    logger.info("Validation: %d violations (%d critical)", len(violations),
                sum(v["type"] in cfg["anomalies"]["critical_types"] for v in violations))
    return violations


def _append_history(stats, path, run_id):
    """Add one summary line per run so trends can be tracked over time."""
    row = {"run_id": run_id, "generated_at": stats["generated_at"],
           "n_rows": stats["n_rows"], "n_participants": stats["n_participants"]}
    row.update({f"label_{k}": v for k, v in stats.get("label_distribution", {}).items()})
    for col in ["L_move_mean", "R_move_mean", "L_tremor_mean", "R_tremor_mean"]:
        if col in stats["columns"]:
            row[f"{col}_mean"] = stats["columns"][col].get("mean", np.nan)
    new = pd.DataFrame([row])
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new.to_csv(path, index=False)
