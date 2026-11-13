import math

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from scripts.core.errors import ConfigError
from scripts.core.metrics import (auc, auc_within, balanced_accuracy, confusion, label_scores, macro_f1, mae,
                                  mean_scores, spearman)


def test_auc_equals_scikit_learn_including_ties():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(2, 200))
        y = rng.random(n) < rng.random()
        score = np.round(rng.random(n), int(rng.integers(1, 4)))        # rounding makes ties
        if 0 < y.sum() < n:
            assert auc(y, score) == pytest.approx(roc_auc_score(y, score), abs=1e-12)


def test_the_class_scores_equal_scikit_learn():
    rng = np.random.default_rng(1)
    for _ in range(50):
        k = int(rng.integers(2, 8))
        n = int(rng.integers(1, 120))
        y = rng.integers(0, k, n)
        predicted = np.where(rng.random(n) < 0.5, y, rng.integers(0, k, n))
        np.testing.assert_array_equal(confusion(y, predicted, k), confusion_matrix(y, predicted, labels=range(k)))
        assert balanced_accuracy(y, predicted, k) == pytest.approx(balanced_accuracy_score(y, predicted))
        assert macro_f1(y, predicted, k) == pytest.approx(f1_score(y, predicted, average="macro", zero_division=0))


def test_spearman_and_mae_equal_scipy_and_numpy():
    rng = np.random.default_rng(2)
    for _ in range(50):
        n = int(rng.integers(3, 100))
        a = rng.integers(-3, 4, n).astype(float)                          # diary scores: many ties
        b = rng.random(n) * 6 - 3
        if len(np.unique(a)) > 1:
            assert spearman(a, b) == pytest.approx(spearmanr(a, b).statistic, abs=1e-12)
        assert mae(a, b) == pytest.approx(np.abs(a - b).mean())
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3])) and math.isnan(mae([], []))


def test_auc_is_nan_without_both_answers_and_refuses_mismatched_inputs():
    assert math.isnan(auc([True, True], [0.1, 0.2])) and math.isnan(auc([], []))
    with pytest.raises(ConfigError, match="same length"):
        auc([True, False], [0.1])


def test_within_averages_each_patients_own_score_skipping_those_with_one_answer():
    groups = ["A"] * 4 + ["B"] * 4 + ["C"] * 2
    y = [1, 0, 1, 0, 1, 1, 0, 0, 1, 1]
    score = [0.9, 0.1, 0.8, 0.2, 0.1, 0.2, 0.9, 0.8, 0.5, 0.5]      # A perfect, B backwards, C one answer
    assert auc_within(groups, y, score) == pytest.approx(0.5)
    assert math.isnan(auc_within(["C", "C"], [1, 1], [0.1, 0.2]))


def test_label_scores_for_the_state():
    probs = np.eye(7)[[0, 3, 5, 1, 3, 2]]                            # one-hot predictions
    y = np.array([0, 3, 5, np.nan, 3, 6])                            # the last hour: +3 predicted as -1
    s = label_scores("state_now", y, probs, ["A", "A", "A", "B", "B", "B"], no_dbs=np.array([1, 1, 1, 0, 0, 0]))
    assert s["hours"] == 5                                            # the NaN hour is left out
    # OFF: the one OFF hour scores 1, but so does the +3 hour predicted as -1: (3 + 0.5) / 4.
    assert s["off_auc"] == pytest.approx(0.875) and s["on_auc"] == 1 and s["off_auc_no_dbs"] == 1
    assert s["recall_severe_akinesia"] == 1 and s["recall_severe_dyskinesia"] == 0
    assert s["precision_slight_akinesia"] == 0 and math.isnan(s["precision_severe_dyskinesia"])   # never predicted
    assert math.isnan(s["recall_slight_dyskinesia"])                 # no hour has that answer
    assert s["score_mae"] == pytest.approx(4 / 5)                    # off by 4 points in one hour of 5
    assert s["balanced_accuracy"] == pytest.approx(3 / 4) and 0 < s["macro_f1"] < 1
    assert s["score_spearman"] == pytest.approx(spearmanr([-3, 0, 2, 0, 3], [-3, 0, 2, 0, -1]).statistic)
    assert {"balanced_accuracy_within", "macro_f1_within", "score_spearman_within", "score_mae_within"} <= set(s)


def test_label_scores_for_tremor_and_sleep():
    tremor = label_scores("tremor_now", [0, 1, 2], np.eye(3), ["A"] * 3)
    assert tremor["any_tremor_auc"] == 1 and tremor["score_spearman"] == 1 and tremor["score_mae"] == 0
    sleep = label_scores("sleep_now", [0, 1], np.eye(2), ["A", "A"])
    assert sleep["asleep_auc"] == 1 and sleep["recall_asleep"] == 1
    assert not any(m.startswith("score_") for m in sleep)              # sleep has no diary score
    with pytest.raises(ConfigError, match="probabilities are"):
        label_scores("sleep_now", [0, 1], np.eye(3), ["A", "A"])


def test_mean_scores_ignores_nan_parts():
    out = mean_scores({1: {"state_now": {"off_auc": 0.6, "x": math.nan}}, 2: {"state_now": {"off_auc": 0.8}}})
    assert out["state_now"]["off_auc"] == pytest.approx(0.7) and math.isnan(out["state_now"]["x"])
