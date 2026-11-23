"""run_model_pipeline --baseline: repeated cross-validation of the hourly baseline, and its report."""

import numpy as np
import pandas as pd
import pytest

from scripts.core.errors import ConfigError
from scripts.run_model_pipeline import run_baseline
from scripts.steps.baseline_cv import question_metrics, settings, summarise
from scripts.tracking.memory_tracker import MemoryTracker
from tests.helpers import make_config
from tests.steps.fake_models import SPLIT, biased_dataset

ENTRY = {"name": "hourly_logistic", "family": "hourly_logistic", "horizons": ["now"], "params": {"c": 1.0}}


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path, baseline={"candidate": ENTRY, "cv_folds": 5, "repeats": 2})


def test_repeats_cross_validation_and_writes_the_report(cfg):
    summary = run_baseline(cfg, MemoryTracker(), data=biased_dataset(), split=SPLIT)
    off = summary.set_index("metric").loc["state_now/off/roc_auc"]
    assert off["repeats"] == 2 and off["mean"] > 0.6 and off["min"] <= off["mean"] <= off["max"]
    assert "state_now/off/pr_auc" in set(summary["metric"]) and "state_now/macro_f1" in set(summary["metric"])
    out = cfg["resolved_paths"]["artifacts_dir"] / "baseline"
    for name in ("repeats.csv", "summary.csv", "summary.json", "coefficients.csv", "report.md", "curves.png"):
        assert (out / name).exists(), name
    assert "ROC AUC" in (out / "report.md").read_text()
    assert len(list((cfg["resolved_paths"]["artifacts_dir"] / "models").iterdir())) == 1   # the final model


def test_the_test_patients_are_never_scored(cfg):
    run_baseline(cfg, MemoryTracker(), data=biased_dataset(), split=SPLIT)
    repeats = pd.read_csv(cfg["resolved_paths"]["artifacts_dir"] / "baseline" / "repeats.csv")
    hours = repeats[repeats["metric"] == "state_now/off/hours"]["value"]
    data = biased_dataset().subset(SPLIT.development)
    answered = data.labels["state_now"].notna().sum()
    assert (hours == answered).all()   # every development hour once per repeat, no test hour


def test_question_metrics_match_a_hand_count():
    y = np.array([1, 1, 0, 0, 0, 0], bool)
    p = np.array([0.9, 0.4, 0.6, 0.1, 0.2, 0.3])
    m = question_metrics(y, p, np.array(["a"] * 6), 0.5)
    assert m["prevalence"] == pytest.approx(1 / 3) and m["roc_auc"] == pytest.approx(7 / 8)
    assert m["sensitivity"] == 0.5 and m["specificity"] == 0.75 and m["precision"] == 0.5
    assert m["pr_lift"] == pytest.approx(m["pr_auc"] * 3)


def test_summary_is_mean_sd_min_max():
    rows = pd.DataFrame({"repeat": [1, 2, 3], "metric": "m", "value": [0.6, 0.7, 0.8]})
    s = summarise(rows).iloc[0]
    assert (s["mean"], s["min"], s["max"]) == pytest.approx((0.7, 0.6, 0.8)) and s["sd"] == pytest.approx(0.1)


def test_settings_are_checked(tmp_path):
    with pytest.raises(ConfigError):
        settings({"baseline": {}})
    with pytest.raises(ConfigError):
        settings({"baseline": {"candidate": ENTRY, "cv_folds": 1}})
