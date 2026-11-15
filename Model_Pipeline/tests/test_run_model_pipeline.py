"""The whole run, every real step in order, on the small fake family and data the bias tests use."""

import datetime as dt
import functools
import json
import tarfile
from types import SimpleNamespace

import pytest

from scripts import run_model_pipeline
from scripts.core.errors import GateError
from scripts.core.tracker import RunStatus
from scripts.models import registry
from scripts.run_model_pipeline import ready_for_selection, run
from scripts.steps import bias, train
from scripts.steps.register import active_version, versions
from scripts.tracking.memory_tracker import MemoryTracker
from tests.helpers import make_config
from tests.steps.fake_models import SPLIT, biased_dataset, pipeline_config

STEPS = ["train", "evaluate", "bias", "explain", "select_model", "compare_production", "export"]
# The release gate's rules that the fake family can be scored on (it predicts now only).
RULES = [{"metric": "cv/state_now/off_auc", "must_improve": True},
         {"metric": "cv/sleep_now/asleep_auc", "must_improve": False}]


@pytest.fixture(autouse=True)
def fake_family(monkeypatch):
    monkeypatch.setitem(registry.FAMILIES, "fake_hourly", "tests.steps.fake_models:HourlyLogistic")


@pytest.fixture(autouse=True)
def ticking_clock(monkeypatch):
    """A clock a second further on at each call: train names each saved model by the second, and
    these tests run the pipeline twice within one."""
    class Ticking(dt.datetime):
        calls = 0

        @classmethod
        def now(cls, tz=None):
            cls.calls += 1
            return dt.datetime(2026, 1, 1, tzinfo=tz) + dt.timedelta(seconds=cls.calls)
    monkeypatch.setattr(train, "dt", SimpleNamespace(datetime=Ticking, timezone=dt.timezone))


@pytest.fixture(autouse=True)
def alerts(monkeypatch):
    """Records alerts instead of sending them, whatever the machine's .env holds."""
    sent = []
    record = lambda subject, body, cfg=None: sent.append((subject, body))   # noqa: E731
    monkeypatch.setattr(run_model_pipeline, "notify", record)
    monkeypatch.setattr(bias, "notify", record)
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    return sent


@pytest.fixture
def cfg(tmp_path):
    """The full config, with the fake family and small bias and explain settings.

    The fake data is biased on purpose; a gap limit of 1 lets a candidate through, so the run
    reaches the release (test_a_biased_model_stops_the_run checks the bias gate). Its tremor is
    noise, so the validation gate checks only the OFF score.
    """
    small = pipeline_config(tmp_path / "artifacts", max_recall_gap=1.0)
    sections = {k: small[k] for k in ("models", "train", "bias", "explain")}
    config = make_config(tmp_path, evaluation={"bootstrap_resamples": 50}, release_gate={"rules": RULES},
                         **sections)
    config["validation_gate"]["min"] = {"cv/state_now/off_auc": 0.5}   # replaced, not merged into
    config["train"].update(use_tuned=False, charts={"enabled": False})  # nothing tuned here
    return config


def run_once(cfg, release=True, tracker=None):
    return run(cfg, tracker or MemoryTracker(), release=release, data=biased_dataset(kind="imbalance"),
               split=SPLIT)


def family_dir(cfg):
    return cfg["resolved_paths"]["registry_dir"] / cfg["export"]["family"]


def test_a_first_run_goes_through_every_step_and_releases_v1(cfg):
    tracker = MemoryTracker()
    assert run_once(cfg, tracker=tracker) == 1
    assert active_version(family_dir(cfg)) == 1
    pipeline = tracker.find("model_pipeline")
    assert pipeline.status == RunStatus.FINISHED and pipeline.tags["split_hash"] == SPLIT.hash
    assert [r.name for r in tracker.children(pipeline.run_id)] == STEPS + ["register"]


def test_the_release_holds_the_chosen_model_its_metrics_and_training_stats(cfg):
    run_once(cfg)
    release = family_dir(cfg) / "v1"
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    with tarfile.open(release / manifest["file"]) as archive:
        assert "./manifest.json" in archive.getnames()                 # the saved model folder, whole
    metrics = json.loads((release / "metrics.json").read_text(encoding="utf-8"))
    assert 0.5 < metrics["cv/state_now/off_auc"] <= 1
    stats = json.loads((release / "stats.json").read_text(encoding="utf-8"))
    assert stats["participants"] == len(SPLIT.development) and "move_mean" in stats["features"]


def test_without_release_every_gate_runs_but_nothing_is_registered(cfg):
    tracker = MemoryTracker()
    assert run_once(cfg, release=False, tracker=tracker) is None
    assert versions(family_dir(cfg)) == []
    assert (cfg["resolved_paths"]["artifacts_dir"] / "bundle" / "manifest.json").is_file()
    pipeline = tracker.find("model_pipeline")
    assert [r.name for r in tracker.children(pipeline.run_id)] == STEPS


def test_a_biased_model_stops_the_run_before_selection(cfg, alerts):
    cfg["bias"]["max_recall_gap"] = 0.01
    tracker = MemoryTracker()
    with pytest.raises(GateError, match="bias gate failed"):
        run_once(cfg, tracker=tracker)
    assert versions(family_dir(cfg)) == []
    assert [r.name for r in tracker.children(tracker.find("model_pipeline").run_id)] == ["train", "evaluate", "bias"]
    assert alerts[0][0] == "[Model pipeline] Bias gate failed"         # P3's own alert, with the gaps


def test_a_model_no_better_than_production_is_not_exported_or_released(cfg):
    run_once(cfg)
    tracker = MemoryTracker()
    with pytest.raises(GateError, match="production gate failed"):
        run_once(cfg, tracker=tracker)
    assert versions(family_dir(cfg)) == [1] and active_version(family_dir(cfg)) == 1
    assert tracker.find("model_pipeline").status == RunStatus.FAILED
    with pytest.raises(Exception, match="expected one run named 'export'"):
        tracker.find("export")


# ---- what joins the steps ---------------------------------------------------------------

def test_only_a_saved_model_is_exportable(tmp_path):
    assert ready_for_selection({"name": "m", "model_dir": str(tmp_path)})["exportable"]
    assert not ready_for_selection({"name": "m", "model_dir": str(tmp_path / "gone")})["exportable"]


# ---- the command and its alerts -----------------------------------------------------------

@pytest.fixture
def main_with(cfg, monkeypatch):
    """main(), with the small config, data and split."""
    monkeypatch.setattr(run_model_pipeline, "load_config", lambda path=None: cfg)
    monkeypatch.setattr(run_model_pipeline, "run", functools.partial(
        run, data=biased_dataset(kind="imbalance"), split=SPLIT))
    return run_model_pipeline.main


def test_main_exits_0_on_release_and_1_on_a_failed_gate(main_with, alerts):
    assert main_with(["--no-mlflow", "--no-register"]) == 0
    assert main_with(["--no-mlflow"]) == 0
    assert main_with(["--no-mlflow"]) == 1
    (checked, _), (released, _), (failed, body) = alerts
    assert checked == "[MedNOVA model] Training finished, every gate passed"
    assert released == "[MedNOVA model] parkinsons-risk v1 released"
    assert failed == "[MedNOVA model] The production gate failed"
    assert "cv/state_now/off_auc" in body and "a local run on" in body


def test_a_crash_is_alerted_with_its_stack_trace_and_re_raised(main_with, alerts, monkeypatch):
    def broken(self, candidates):
        raise KeyError("missing column")
    monkeypatch.setattr(run_model_pipeline.EvaluateModel, "run", broken)
    with pytest.raises(KeyError):
        main_with(["--no-mlflow"])
    (subject, body), = alerts
    assert subject == "[MedNOVA model] Run crashed: KeyError"
    assert "Traceback" in body and "missing column" in body


def test_alerts_link_to_the_github_actions_run(monkeypatch):
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "MLOps-Northeastern-University/MLModel")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    assert run_model_pipeline.where() == "https://github.com/MLOps-Northeastern-University/MLModel/actions/runs/42"
