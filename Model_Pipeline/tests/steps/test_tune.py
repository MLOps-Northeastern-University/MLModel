import json

import pandas as pd
import pytest

from scripts.core.errors import ConfigError
from scripts.data.folds import Split
from scripts.steps.train import TrainModels
from scripts.steps.tune import TuneModels, nest
from scripts.tracking.memory_tracker import MemoryTracker
from tests.models.test_window_classifier import environment  # noqa: F401  (the fake harnet10)
from tests.steps.test_train import SPLIT, TINY_TRANSFORMER, WINDOWS, data

LOGISTIC = {"name": "window_logistic", "family": "window_logistic"}


def config(tmp_path, tune, entries=(LOGISTIC,), metric="cv/state_now/off_auc_within"):
    return {"windows": WINDOWS, "resolved_paths": {"artifacts_dir": tmp_path},
            "models": {"targets": ["sleep", "state", "tremor"], "horizons": ["now", "next_hour"],
                       "candidates": list(entries)},
            "selection": {"metric": metric},
            "train": {"seeds": [0], "score_test": False, "use_tuned": True, "leakage_check": {"candidates": []}},
            "tune": {"seed": 0, "candidates": tune}}


def logistic(trials, space=None):
    return {"window_logistic": {"trials": trials,
                                "space": space if space is not None else
                                {"c": {"type": "float", "low": 1e-3, "high": 1.0, "log": True}}}}


def test_each_trial_is_a_tracked_cross_validation_and_the_best_is_kept(tmp_path):
    tracker = MemoryTracker()
    out = TuneModels(config(tmp_path, logistic(3)), tracker).run(data(), SPLIT)
    entry = out["window_logistic"]
    assert entry["trials"] == 3 and 1e-3 <= entry["params"]["c"] <= 1.0 and entry["split_hash"] == SPLIT.hash
    tune = tracker.find("tune")
    assert tune.tags["metric"] == "cv/state_now/off_auc_within"
    [candidate] = tracker.children(tune.run_id)
    trials = tracker.children(candidate.run_id)
    assert [t.name for t in trials] == ["trial-0", "trial-1", "trial-2"]
    assert [f.name for f in tracker.children(trials[0].run_id)] == ["seed-0-fold-1", "seed-0-fold-2", "seed-0-fold-3"]
    values = [t.latest("cv/state_now/off_auc_within") for t in trials]
    assert entry["value"] == max(values) and candidate.latest("run/tune_trials") == 3
    assert candidate.params["best.c"] == entry["params"]["c"]
    assert candidate.artifacts and candidate.artifacts[0][1] == "tuning"
    table = pd.read_csv(tmp_path / "tuning" / "window_logistic.csv")
    assert list(table.state) == ["COMPLETE"] * 3 and "param.c" in table.columns
    saved = json.loads((tmp_path / "tuned_params.json").read_text(encoding="utf-8"))
    assert saved["window_logistic"]["params"] == entry["params"]


def test_a_rerun_resumes_the_study_and_a_change_starts_a_new_one(tmp_path):
    TuneModels(config(tmp_path, logistic(2)), MemoryTracker()).run(data(), SPLIT)
    tracker = MemoryTracker()
    out = TuneModels(config(tmp_path, logistic(4)), tracker).run(data(), SPLIT)      # 2 more, not 4
    assert out["window_logistic"]["trials"] == 4
    assert [r.name for r in tracker.runs.values() if r.name.startswith("trial-")] == ["trial-2", "trial-3"]
    other_space = logistic(1, {"c": {"type": "float", "low": 0.01, "high": 0.1}})
    tracker = MemoryTracker()
    out = TuneModels(config(tmp_path, other_space), tracker).run(data(), SPLIT)
    assert out["window_logistic"]["trials"] == 1 and tracker.find("trial-0")         # a new study


def test_settings_the_model_refuses_fail_that_trial_only(tmp_path):
    space = {"c": {"type": "categorical", "choices": [-1.0, 0.05]}}
    out = TuneModels(config(tmp_path, logistic(4, space)), MemoryTracker()).run(data(), SPLIT)
    table = pd.read_csv(tmp_path / "tuning" / "window_logistic.csv")
    assert set(table.state) <= {"COMPLETE", "FAIL"} and out["window_logistic"]["params"]["c"] == 0.05


def test_train_uses_the_tuned_settings(tmp_path):
    out = TuneModels(config(tmp_path, logistic(2)), MemoryTracker()).run(data(), SPLIT)
    tracker = MemoryTracker()
    [result] = TrainModels(config(tmp_path, {}), tracker).run(data(), SPLIT)
    assert result["tuned"] is True
    assert tracker.runs[result["run_id"]].params["params.c"] == out["window_logistic"]["params"]["c"]


def test_settings_tuned_on_another_split_are_reused_and_marked_stale(tmp_path):
    out = TuneModels(config(tmp_path, logistic(2)), MemoryTracker()).run(data(), SPLIT)
    other = Split(["G"], [["A", "C"], ["B", "D"], ["E", "F"]], seed=0)
    tracker = MemoryTracker()
    [result] = TrainModels(config(tmp_path, {}), tracker).run(data(), other)
    assert result["tuned"] is True
    assert tracker.runs[result["run_id"]].params["params.c"] == out["window_logistic"]["params"]["c"]
    assert tracker.find("train").params["tuned.window_logistic.stale"] == ["split"]


def test_a_candidate_whose_config_changed_trains_with_its_config(tmp_path):
    TuneModels(config(tmp_path, logistic(2)), MemoryTracker()).run(data(), SPLIT)
    changed = {**LOGISTIC, "params": {"max_iter": 100}}
    [result] = TrainModels(config(tmp_path, {}, entries=(changed,)), MemoryTracker()).run(data(), SPLIT)
    assert result["tuned"] is False


def test_with_nothing_tuned_yet_every_candidate_trains_with_its_config(tmp_path):
    [result] = TrainModels(config(tmp_path, {}), MemoryTracker()).run(data(), SPLIT)
    assert result["tuned"] is False


def test_tuning_the_transformers_diary_step_pretrains_each_fold_once(tmp_path):
    entry = {"name": "mct", "family": "motor_causal_transformer", "params": TINY_TRANSFORMER}
    tune = {"mct": {"trials": 2, "space": {"diary.lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}}}}
    out = TuneModels(config(tmp_path, tune, entries=(entry,)), MemoryTracker()).run(data(), SPLIT)
    assert set(out["mct"]["params"]) == {"diary"} and set(out["mct"]["params"]["diary"]) == {"lr"}
    assert len(list((tmp_path / "pretrained_cache").glob("*.pt"))) == 3      # one per fold, for both trials


def test_nothing_listed_means_nothing_tuned(tmp_path):
    assert TuneModels(config(tmp_path, {}), MemoryTracker()).run(data(), SPLIT) == {}


def test_nest_turns_dotted_names_into_blocks():
    assert nest({"diary.lr": 1, "diary.weight_decay": 2, "c": 3}) == {"diary": {"lr": 1, "weight_decay": 2}, "c": 3}


@pytest.mark.parametrize("tune, match", [
    ({"svm": {"trials": 2, "space": {"c": {"type": "float", "low": 0, "high": 1}}}}, "unknown candidate"),
    (logistic(0), "trials must be"),
    (logistic(2, {}), "at least one setting"),
    (logistic(2, {"depth": {"type": "int", "low": 1, "high": 3}}), "has no setting 'depth'"),
    (logistic(2, {"c": {"type": "float", "low": 1.0, "high": 0.1}}), "low < high"),
    (logistic(2, {"c": {"type": "float", "low": 0.0, "high": 1.0, "log": True}}), "log scale"),
    (logistic(2, {"c": {"type": "normal"}}), "type must be"),
    (logistic(2, {"c": {"type": "categorical", "choices": []}}), "choices"),
    (logistic(2, {"max_iter": {"type": "int", "low": 1.5, "high": 9}}), "low < high"),
])
def test_bad_tuning_config_is_refused_before_anything_trains(tmp_path, tune, match):
    with pytest.raises(ConfigError, match=match):
        TuneModels(config(tmp_path, tune), MemoryTracker())


def test_the_metric_is_selections_and_must_be_a_cv_score(tmp_path):
    with pytest.raises(ConfigError, match="cross-validation score"):
        TuneModels(config(tmp_path, logistic(2), metric="val/state_now/off_auc"), MemoryTracker())
    with pytest.raises(ConfigError, match="selection.metric"):
        TuneModels(config(tmp_path, logistic(2), metric="off"), MemoryTracker())


def test_a_block_of_the_transformer_settings_cant_be_tuned_whole(tmp_path):
    entry = {"name": "mct", "family": "motor_causal_transformer", "params": TINY_TRANSFORMER}
    tune = {"mct": {"trials": 1, "space": {"diary": {"type": "float", "low": 0.1, "high": 1}}}}
    with pytest.raises(ConfigError, match="block of settings"):
        TuneModels(config(tmp_path, tune, entries=(entry,)), MemoryTracker())


def test_an_error_metric_is_minimised(tmp_path):
    tracker = MemoryTracker()
    out = TuneModels(config(tmp_path, logistic(3), metric="cv/state_now/score_mae"), tracker).run(data(), SPLIT)
    values = [r.latest("cv/state_now/score_mae") for r in tracker.runs.values() if r.name.startswith("trial-")]
    assert out["window_logistic"]["value"] == min(values)
    with pytest.raises(ConfigError, match="not a count or a spread"):
        TuneModels(config(tmp_path, logistic(2), metric="cv/state_now/off_auc_fold_sd"), MemoryTracker())
