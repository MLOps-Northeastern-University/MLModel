import math

import numpy as np
import pandas as pd
import pytest

from scripts.core.targets import STATE_CLASSES
from scripts.reporting import charts


def state_frame(y, predicted, seeds=(0,), worn=None):
    """Prediction rows for state_now: each hour's most likely class is `predicted`."""
    parts = []
    for seed in seeds:
        probs = np.full((len(y), 7), 0.01)
        probs[np.arange(len(y)), predicted] = 0.94
        frame = pd.DataFrame({"participant_id": "A", "hour_index": np.arange(len(y)), "fold": "1", "seed": seed,
                              "model": "m", "worn": True if worn is None else worn, "no_dbs": True,
                              "y_true": np.asarray(y, dtype=float)})
        parts.append(pd.concat([frame, pd.DataFrame(probs, columns=list(STATE_CLASSES))], axis=1))
    return pd.concat(parts, ignore_index=True)


def test_the_confusion_matrix_counts_only_scored_hours_and_knows_the_seeds():
    y = [0, 0, 3, 3, np.nan, 6]
    predicted = [0, 3, 3, 3, 3, 3]
    worn = np.array([True, True, True, True, True, False])
    counts, names, seeds = charts.confusion_counts("state_now", state_frame(y, predicted, worn=worn))
    assert names == ["-3", "-2", "-1", "0", "+1", "+2", "+3"] and seeds == 1
    assert counts[0, 0] == 1 and counts[0, 3] == 1 and counts[3, 3] == 2 and counts.sum() == 4
    counts, _, seeds = charts.confusion_counts("state_now", state_frame(y, predicted, seeds=(0, 1), worn=worn))
    assert seeds == 2 and counts.sum() == 8


def test_class_names_are_the_diary_scores_where_there_are_some():
    assert charts.class_names("sleep_now") == ["awake", "asleep"]
    assert charts.class_names("tremor_next_hour") == ["0", "1", "2"]


def test_chance_is_where_guessing_lands():
    assert charts.chance_level("state_now", "off_auc") == 0.5
    assert charts.chance_level("state_now", "balanced_accuracy") == pytest.approx(1 / 7)
    assert charts.chance_level("sleep_now", "balanced_accuracy") == 0.5
    assert charts.chance_level("state_now", "score_spearman") == 0.0
    assert charts.chance_level("state_now", "macro_f1") is None
    assert charts.chance_level("state_now", "off_auc_within") == 0.5


def test_the_scorecard_shows_each_question_pooled_and_within_and_skips_what_is_missing():
    scores = {"off_auc": 0.7, "off_auc_within": 0.8, "on_auc": math.nan, "on_auc_within": math.nan,
              "balanced_accuracy": 0.2, "macro_f1": 0.1, "score_spearman": -0.1, "score_spearman_within": 0.3}
    rows = charts.scorecard_rows("state_now", scores)
    assert [r[0] for r in rows] == ["off_auc", "balanced_accuracy", "macro_f1", "score_spearman"]
    assert rows[0][1:] == (0.7, 0.8) and math.isnan(rows[1][2])
    figure = charts.scorecard_figure({"state_now": {**scores, "hours": 12}, "sleep_now": {"asleep_auc": 0.9}}, "t")
    assert len([ax for ax in figure.axes if ax.get_visible()]) == 2
    charts.pyplot().close(figure)


def test_a_scorecard_with_nothing_to_draw_says_so():
    with pytest.raises(ValueError, match="no scores"):
        charts.scorecard_figure({"state_now": {"hours": 3}}, "t")


def test_the_comparison_draws_a_panel_per_scored_metric_and_marks_the_best():
    results = {"a": {"cv/state_now/off_auc": 0.6, "cv/state_now/off_auc_fold_sd": 0.05, "cv/state_now/score_mae": 1.0},
               "b": {"cv/state_now/off_auc": 0.7, "cv/state_now/score_mae": 0.8}}
    figure = charts.comparison_figure(results, ["cv/state_now/off_auc", "cv/state_now/score_mae", "cv/x/never"], "t")
    shown = [ax for ax in figure.axes if ax.get_visible()]
    assert [ax.get_title() for ax in shown] == ["cv/state_now/off_auc", "cv/state_now/score_mae  (lower is better)"]
    bars = shown[1].patches                         # b has the lower error: b is the dark (best) bar
    assert bars[1].get_facecolor() != bars[0].get_facecolor()
    charts.pyplot().close(figure)
    with pytest.raises(ValueError, match="were scored"):
        charts.comparison_figure(results, ["cv/x/never"], "t")
