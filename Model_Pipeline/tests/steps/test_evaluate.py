import json
import math

import numpy as np
import pandas as pd
import pytest

from model_pipeline.core.targets import get_target
from model_pipeline.steps.evaluate import EvaluateModel, evaluate_candidate, gate_failures, scored_rows
from model_pipeline.tracking.memory_tracker import MemoryTracker
from tests.helpers import make_config

PEOPLE = [f"COPS-{i}" for i in range(1, 7)]


def predictions(how="right", model="m", seeds=(0,), hours=24):
    """{label: rows} as train writes them: right puts 0.9 on the true class, wrong on the opposite end."""
    rng = np.random.default_rng(0)
    out = {}
    for label, target in [("state_now", "state"), ("tremor_now", "tremor"), ("sleep_now", "sleep")]:
        classes = get_target(target).classes
        parts = []
        for seed in seeds:
            for i, p in enumerate(PEOPLE):
                y = rng.integers(0, len(classes), hours)
                y[:len(classes)] = np.arange(len(classes))   # every patient has every class
                hot = y if how == "right" else len(classes) - 1 - y
                probs = np.full((hours, len(classes)), 0.1 / (len(classes) - 1))
                probs[np.arange(hours), hot] = 0.9
                rows = pd.DataFrame({"participant_id": p, "hour_index": np.arange(hours), "fold": i // 2 + 1,
                                     "seed": seed, "model": model, "worn": True, "y_true": y.astype(float)})
                parts.append(pd.concat([rows, pd.DataFrame(probs, columns=list(classes))], axis=1))
        out[label] = pd.concat(parts, ignore_index=True)
    return out


def write_run(cfg, frames, metrics, name="m"):
    folder = cfg["resolved_paths"]["artifacts_dir"]
    (folder / "predictions").mkdir(parents=True, exist_ok=True)
    for label, rows in frames.items():
        rows.to_csv(folder / "predictions" / f"{label}.csv", index=False)
    candidates = [{"name": name, "family": "f", "run_id": "r1", "metrics": metrics}]
    (folder / "candidates.json").write_text(json.dumps(candidates))
    return candidates


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path, evaluation={"bootstrap_resamples": 50})


PASSING = {"cv/state_now/off_auc": 0.8, "cv/tremor_now/any_tremor_auc": 0.7}


def test_perfect_predictions_have_an_interval_at_1(cfg):
    extra, details = evaluate_candidate(predictions(), cfg)
    for name in ("cv/state_now/off_auc", "cv/state_now/dyskinetic_auc", "cv/tremor_now/any_tremor_auc",
                 "cv/sleep_now/asleep_auc"):
        assert extra[f"{name}_lo"] == pytest.approx(1.0) and extra[f"{name}_hi"] == pytest.approx(1.0)
    assert set(details["state_now/off_auc"]["folds"]) == {"1", "2", "3"}
    assert set(details["state_now/off_auc"]["patients"]) == set(PEOPLE)


def test_wrong_predictions_have_an_interval_below_chance(cfg):
    extra, _ = evaluate_candidate(predictions("wrong"), cfg)
    assert extra["cv/state_now/off_auc_hi"] < 0.5


def test_only_the_hours_train_scores_count(cfg):
    frames = predictions()
    state = frames["state_now"]
    extra_rows = state.iloc[:5].assign(fold="test")   # the test patients' one scoring
    unworn = state.iloc[:5].assign(worn=False)
    other = state.iloc[:5].assign(model="other")
    frames["state_now"] = pd.concat([state, extra_rows, unworn, other], ignore_index=True)
    frames["state_now"].loc[0, "y_true"] = np.nan
    write_run(cfg, frames, PASSING)
    rows = scored_rows(cfg["resolved_paths"]["artifacts_dir"] / "predictions", "m")
    assert len(rows["state_now"]) == len(state) - 1


def test_seeds_are_scored_apart_and_averaged(cfg):
    extra, _ = evaluate_candidate(predictions(seeds=(0, 1)), cfg)
    assert extra["cv/state_now/off_auc_lo"] == pytest.approx(1.0)


def test_the_gate_fails_on_a_low_or_missing_metric():
    minimums = {"cv/state_now/off_auc": 0.5, "cv/tremor_now/any_tremor_auc": 0.5}
    assert gate_failures(PASSING, minimums) == []
    assert gate_failures({"cv/state_now/off_auc": 0.4, "cv/tremor_now/any_tremor_auc": math.nan}, minimums) == [
        "cv/state_now/off_auc: 0.400 < 0.5", "cv/tremor_now/any_tremor_auc: missing"]


def test_the_step_writes_the_files_and_passes_a_good_candidate(cfg):
    write_run(cfg, predictions(), PASSING)
    tracker = MemoryTracker()
    [c] = EvaluateModel(cfg, tracker).run()
    assert c["gates_passed"] and c["metrics"]["cv/state_now/off_auc"] == 0.8
    folder = cfg["resolved_paths"]["artifacts_dir"] / "evaluation" / "m"
    written = json.loads((folder / "metrics.json").read_text())
    assert written["cv/state_now/off_auc_lo"] == pytest.approx(1.0) and "cv/state_now/off_auc" in written
    assert (folder / "evaluation.json").exists() and (folder / "roc.png").exists()
    assert tracker.find("m").metrics["cv/state_now/off_auc_hi"] == [(None, 1.0)]


def test_a_candidate_below_the_gate_is_marked_failed_not_raised(cfg):
    candidates = write_run(cfg, predictions(), {"cv/state_now/off_auc": 0.45})
    [c] = EvaluateModel(cfg, MemoryTracker()).run(candidates)
    assert not c["gates_passed"]
    assert c["validation_failures"] == ["cv/state_now/off_auc: 0.450 < 0.5", "cv/tremor_now/any_tremor_auc: missing"]
