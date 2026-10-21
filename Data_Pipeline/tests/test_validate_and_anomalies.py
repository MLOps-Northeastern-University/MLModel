import numpy as np
import pandas as pd
import pytest

from scripts.anomalies import detect_anomalies, format_report
from scripts.validate import check_expectations, infer_schema, run_validation, validate_against_schema
from tests.helpers import make_config


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path)


def good_df(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "participant_id": ["COPS-1"] * (n // 2) + ["COPS-2"] * (n - n // 2),
        "hour_index": list(range(n // 2)) * 2,
        "kinesia_class": rng.choice(["off", "on", "dyskinetic"], n),
        "kinesia_score": rng.integers(-3, 4, n),
        "L_move_mean": rng.uniform(0.01, 0.1, n),
        "R_move_mean": rng.uniform(0.01, 0.1, n),
        "L_still_fraction": rng.uniform(0, 1, n),
    })


def test_good_data_passes_expectations(cfg):
    assert check_expectations(good_df(), cfg) == []


def test_bad_label_and_fraction_fail(cfg):
    df = good_df()
    df.loc[0, "kinesia_class"] = "unknown"
    df.loc[1, "L_still_fraction"] = 1.5
    names = {f["expectation"] for f in check_expectations(df, cfg)}
    assert {"kinesia_class_in_set", "L_still_fraction_between_0_and_1"} <= names


def test_schema_detects_missing_column_and_drift(cfg):
    schema = infer_schema(good_df(), cfg)
    new = good_df(seed=1).drop(columns=["R_move_mean"])
    new["L_move_mean"] = new["L_move_mean"] * 100
    types = {(v["type"], v["column"]) for v in validate_against_schema(new, schema, cfg)}
    assert ("schema_violation", "R_move_mean") in types
    assert ("schema_warning", "L_move_mean") in types


def test_run_validation_saves_baseline_and_history(cfg):
    run_validation(good_df(), cfg, "run1")
    run_validation(good_df(seed=2), cfg, "run2")
    reports = cfg["resolved_paths"]["reports_dir"]
    assert (reports / "schema_baseline.json").exists()
    assert len(pd.read_csv(reports / "stats_history.csv")) == 2


def test_missing_values_and_outliers_detected(cfg):
    df = good_df(200)
    df.loc[:20, "L_move_mean"] = np.nan
    df.loc[30:40, "R_move_mean"] = 1000.0
    report = detect_anomalies(df, cfg)
    types = {a["type"] for a in report["warnings"]}
    assert {"missing_values", "outliers"} <= types


def test_empty_dataset_is_critical(cfg):
    report = detect_anomalies(good_df().iloc[0:0], cfg)
    assert report["critical"][0]["type"] == "empty_dataset"
    assert "CRITICAL" in format_report(report)
