import pytest

from model_pipeline.core.errors import ConfigError
from model_pipeline.reporting import charts as charts_module
from model_pipeline.reporting.writer import DEFAULT_COMPARE, ChartWriter
from model_pipeline.tracking.memory_tracker import MemoryTracker
from tests.reporting.test_charts import state_frame


def writer(tmp_path, tracker, **kw):
    return ChartWriter(tmp_path / "charts", tracker, **kw)


def scores():
    return {"state_now": {"hours": 4, "off_auc": 0.7, "off_auc_within": 0.8, "balanced_accuracy": 0.3}}


def test_each_run_gets_its_own_scorecard_and_confusion_matrices(tmp_path):
    tracker = MemoryTracker()
    w = writer(tmp_path, tracker)
    with tracker.run("fold"):
        folder = w.predictions({"state_now": state_frame([0, 3, 3, 6], [0, 3, 3, 3])}, scores(), "m/seed-0-fold-1", "t")
    assert folder == tmp_path / "charts" / "m" / "seed-0-fold-1"
    assert sorted(p.name for p in folder.iterdir()) == ["confusion_state_now.png", "scorecard.png"]
    assert tracker.find("fold").artifacts == [(folder, "charts")]


def test_a_chart_that_fails_is_a_warning_and_the_others_are_still_saved(tmp_path, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("no font")

    monkeypatch.setattr(charts_module, "confusion_figure", broken)
    tracker = MemoryTracker()
    with tracker.run("fold"):
        folder = writer(tmp_path, tracker).predictions({"state_now": state_frame([0], [0])}, scores(), "m/cv", "t")
    assert [p.name for p in folder.iterdir()] == ["scorecard.png"]
    assert "confusion_state_now" in caplog.text and "no font" in caplog.text


def test_nothing_drawn_attaches_nothing(tmp_path):
    tracker = MemoryTracker()
    with tracker.run("fold"):
        assert writer(tmp_path, tracker).predictions({}, {"state_now": {"hours": 0}}, "m/cv", "t") is None
    assert tracker.find("fold").artifacts == []


def test_folder_names_stay_inside_the_charts_folder(tmp_path):
    tracker = MemoryTracker()
    with tracker.run("r"):
        folder = writer(tmp_path, tracker).comparison({"a": {"cv/state_now/off_auc": 0.6}}, where="../../x y")
        assert folder == tmp_path / "charts" / "x_y"
        with pytest.raises(ValueError, match="folder of their own"):
            writer(tmp_path, tracker).comparison({"a": {"cv/state_now/off_auc": 0.6}}, where="..")


def test_config_turns_charts_off_or_on_and_refuses_bad_settings(tmp_path):
    base = {"resolved_paths": {"artifacts_dir": tmp_path}}
    on = ChartWriter.from_config({**base, "train": {}}, MemoryTracker())
    assert on.folds and on.compare == DEFAULT_COMPARE and on.root == tmp_path / "charts"
    assert ChartWriter.from_config({**base, "train": {"charts": {"enabled": False}}}, MemoryTracker()) is None
    for bad in ({"enabled": "yes"}, {"compare": []}, {"compare": "cv/a/b"}, {"colour": 1}):
        with pytest.raises(ConfigError, match="train.charts"):
            ChartWriter.from_config({**base, "train": {"charts": bad}}, MemoryTracker())
