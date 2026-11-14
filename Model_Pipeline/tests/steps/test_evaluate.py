import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from scripts.core.data import WindowSpec
from scripts.core.errors import GateError, SchemaError
from scripts.core.targets import build_labels, get_target
from scripts.steps.evaluate import EvaluateModel, gate_failures, prediction_columns, score, tracking
from scripts.tracking.memory_tracker import MemoryTracker
from tests.helpers import make_config, make_dataset

GATE = ["cv/state_now/off_auc", "cv/state_now/off_auc_no_dbs", "cv/tremor_now/auc",
        "cv/state_now/dyskinetic_auc", "cv/state_now/asleep_auc", "cv/state_next_hour/off_auc"]


@pytest.fixture
def data():
    data = make_dataset(WindowSpec(1, 10, 30, 3), n_participants=4, hours=24)
    hours = data.hours.assign(kinesia_text=np.where(data.hours["hour_index"] % 6 == 0, "Sleep", "Good Kinesia"))
    hours.loc[hours["kinesia_text"] == "Sleep", "kinesia_score"] = np.nan
    return replace(data, hours=hours, labels=build_labels(hours, ["state", "tremor"]))


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path, evaluation={"bootstrap_resamples": 50})


def predictions(data, how="right", seed=None):
    """Predictions for every label: right puts 0.9 on the true class, wrong on another."""
    rng = np.random.default_rng(0)
    rows = []
    for label in data.labels.columns:
        classes = get_target(label.split("_")[0]).classes
        truth = data.labels[label].fillna(0).astype(int).to_numpy()
        hot = truth if how == "right" else (truth + 1 + rng.integers(0, len(classes) - 1, len(truth))) % len(classes)
        p = np.full((len(truth), len(classes)), 0.1 / (len(classes) - 1))
        p[np.arange(len(truth)), hot] = 0.9
        frame = pd.DataFrame(p, columns=prediction_columns(label.split("_")[0]))
        frame.insert(0, "label", label)
        frame.insert(0, "fold", data.hours["participant_id"].map(lambda pid: (int(pid[5:]) + 1) // 2).to_numpy())
        frame.insert(0, "hour_index", data.hours["hour_index"].to_numpy())
        frame.insert(0, "participant_id", data.hours["participant_id"].to_numpy())
        if seed is not None:
            frame["seed"] = seed
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def test_perfect_predictions_score_1_on_every_gate_metric(data, cfg):
    metrics, _ = score(predictions(data), data, cfg)
    for name in GATE:
        assert metrics[name] == pytest.approx(1.0), name
    assert metrics["cv/state_now/off_auc_folds"] == 2
    assert metrics["cv/state_now/pooled_off_auc_lo"] == pytest.approx(1.0)


def test_wrong_predictions_score_below_chance(data, cfg):
    metrics, _ = score(predictions(data, "wrong"), data, cfg)
    assert metrics["cv/state_now/off_auc"] < 0.5 and metrics["cv/tremor_now/auc"] < 0.5


def test_no_dbs_uses_only_participants_without_dbs(data, cfg):
    _, details = score(predictions(data), data, cfg)
    participants = details["seeds"]["None"]["state_now/off_auc"]["participants"]
    assert set(participants) == {"COPS-1", "COPS-2", "COPS-3", "COPS-4"}
    # COPS-2 and COPS-4 have no DBS: one per fold.
    assert set(details["seeds"]["None"]["state_now/off_auc"]["folds_no_dbs"]) == {"1", "2"}


def test_a_fold_with_one_class_is_left_out_and_counted(data, cfg):
    data.labels.loc[data.hours["participant_id"].isin(["COPS-1", "COPS-2"]), "state_now"] = 0.0  # fold 1 all OFF
    metrics, _ = score(predictions(data), data, cfg)
    assert metrics["cv/state_now/off_auc_folds"] == 1 and metrics["cv/state_now/off_auc"] == pytest.approx(1.0)


def test_unlabelled_hours_are_left_out(data, cfg):
    data.labels.loc[:5, "tremor_now"] = np.nan
    preds = predictions(data)
    metrics, _ = score(preds, data, cfg)
    assert metrics["cv/tremor_now/auc"] == pytest.approx(1.0)


def test_tracking_follows_the_diary_within_each_participant():
    rows = pd.DataFrame({"participant_id": ["A"] * 4 + ["B"] * 4, "y_true": [0, 1, 2, 2, 1, 1, 1, 1],
                         "p_none": [0.8, 0.5, 0.1, 0.2, 0.3, 0.3, 0.3, 0.3],
                         "p_weak_or_slight": [0.1, 0.4, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4],
                         "p_strong_or_severe": [0.1, 0.1, 0.7, 0.6, 0.3, 0.3, 0.3, 0.3]})
    assert tracking(rows, "tremor") == {"A": pytest.approx(rows.iloc[:4].assign(
        e=lambda r: r.p_weak_or_slight + 2 * r.p_strong_or_severe)["e"].corr(rows.y_true[:4], method="spearman"))}


def test_several_seeds_give_the_mean_and_spread(data, cfg):
    both = pd.concat([predictions(data, seed=1), predictions(data, "wrong", seed=2)], ignore_index=True)
    metrics, details = score(both, data, cfg)
    one, _ = score(predictions(data, "wrong"), data, cfg)
    assert metrics["cv/state_now/off_auc"] == pytest.approx((1 + one["cv/state_now/off_auc"]) / 2)
    assert metrics["cv/state_now/off_auc_seed_std"] > 0 and set(details["seeds"]) == {"1", "2"}


@pytest.mark.parametrize("change, match", [
    (lambda p: p.drop(columns="fold"), "fold"),
    (lambda p: p.drop(columns="p_asleep"), "p_asleep"),
    (lambda p: p.assign(p_none=p["p_none"] + 0.5), "add up to 1"),
    (lambda p: p.assign(hour_index=p["hour_index"] + 1000), "aren't in the dataset"),
])
def test_bad_predictions_are_refused(data, cfg, change, match):
    with pytest.raises(SchemaError, match=match):
        score(change(predictions(data)), data, cfg)


def test_the_gate_fails_on_a_low_or_missing_metric():
    minimums = {"cv/state_now/off_auc": 0.5, "cv/tremor_now/auc": 0.5}
    assert gate_failures({"cv/state_now/off_auc": 0.7, "cv/tremor_now/auc": 0.6}, minimums) == []
    assert gate_failures({"cv/state_now/off_auc": 0.4, "cv/tremor_now/auc": math.nan}, minimums) == [
        "cv/state_now/off_auc: 0.400 < 0.5", "cv/tremor_now/auc: missing"]


def test_the_step_writes_the_files_logs_and_passes(data, cfg):
    tracker = MemoryTracker()
    metrics = EvaluateModel(cfg, tracker).run(predictions(data), data)
    folder = cfg["resolved_paths"]["artifacts_dir"]
    written = json.loads((folder / "metrics.json").read_text())
    assert set(GATE) <= set(written) and written["cv/state_now/off_auc"] == metrics["cv/state_now/off_auc"]
    assert (folder / "evaluation.json").exists() and (folder / "roc.png").exists()
    run = tracker.find("evaluate")
    assert run.tags["validation_gate"] == "passed" and run.metrics["cv/tremor_now/auc"] == [(None, 1.0)]


def test_the_step_raises_when_the_gate_fails(data, cfg):
    tracker = MemoryTracker()
    with pytest.raises(GateError, match="validation"):
        EvaluateModel(cfg, tracker).run(predictions(data, "wrong"), data)
    assert tracker.find("evaluate").tags["validation_gate"] == "failed"
