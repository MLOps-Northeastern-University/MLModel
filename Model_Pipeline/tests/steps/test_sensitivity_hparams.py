import json

import pytest

from scripts.core.errors import ConfigError
from scripts.steps.sensitivity_hparams import HyperparameterSensitivity
from scripts.steps.tune import TuneModels
from scripts.tracking.memory_tracker import MemoryTracker
from tests.models.test_window_classifier import environment  # noqa: F401  (the fake harnet10)
from tests.steps.test_train import SPLIT, data
from tests.steps.test_tune import config, logistic

SPACE = {"c": {"type": "float", "low": 1e-4, "high": 1.0, "log": True},
         "class_weight_power": {"type": "float", "low": 0.0, "high": 1.0}}


def test_importance_and_plots_for_each_tuned_candidate(tmp_path):
    # The synthetic movement shows the state perfectly, so AUCs don't move; the score's error does.
    cfg = config(tmp_path, logistic(5, SPACE), metric="cv/state_now/score_mae")
    TuneModels(cfg, MemoryTracker()).run(data(), SPLIT)
    tracker = MemoryTracker()
    out = HyperparameterSensitivity(cfg, tracker).run()
    result = out["window_logistic"]
    assert result["trials"] == 5 and set(result["importance"]) == {"c", "class_weight_power"}
    assert sum(result["importance"].values()) == pytest.approx(1.0)
    folder = tmp_path / "sensitivity" / "window_logistic"
    assert {p.name for p in folder.iterdir()} == {"importance.json", "importance.png", "slice.png", "history.png"}
    assert all((folder / f).stat().st_size > 1000 for f in ("importance.png", "slice.png", "history.png"))
    saved = json.loads((folder / "importance.json").read_text(encoding="utf-8"))
    assert saved["importance"] == result["importance"] and saved["metric"] == "cv/state_now/score_mae"
    assert saved["flat"] is False
    run = tracker.find("window_logistic")
    assert run.latest("run/importance_c") == pytest.approx(result["importance"]["c"])
    assert run.artifacts == [(folder, "sensitivity")]


def test_when_no_setting_changes_the_score_every_importance_is_zero(tmp_path):
    cfg = config(tmp_path, logistic(3, SPACE))                        # AUC 1 in every trial
    TuneModels(cfg, MemoryTracker()).run(data(), SPLIT)
    result = HyperparameterSensitivity(cfg, MemoryTracker()).run()["window_logistic"]
    assert result["importance"] == {"c": 0.0, "class_weight_power": 0.0}
    folder = tmp_path / "sensitivity" / "window_logistic"
    assert json.loads((folder / "importance.json").read_text(encoding="utf-8"))["flat"] is True
    assert not (folder / "importance.png").exists() and (folder / "slice.png").exists()


def test_too_few_trials_are_skipped_not_guessed(tmp_path):
    cfg = config(tmp_path, logistic(2, SPACE))
    TuneModels(cfg, MemoryTracker()).run(data(), SPLIT)
    assert HyperparameterSensitivity(cfg, MemoryTracker()).run() == {}


def test_it_needs_tuning_first(tmp_path):
    with pytest.raises(ConfigError, match="run tune before"):
        HyperparameterSensitivity(config(tmp_path, {}), MemoryTracker()).run()
