"""The whole run, every real step in order, on the small fake family and data the bias tests use."""

import functools
import json

import pytest

from model_pipeline import run_model_pipeline
from model_pipeline.core.errors import GateError
from model_pipeline.core.tracker import RunStatus
from model_pipeline.run_model_pipeline import run
from model_pipeline.steps import bias
from model_pipeline.steps.evaluate import EvaluateModel
from model_pipeline.steps.register import active_version, versions
from model_pipeline.steps.tune import TuneModels
from model_pipeline.tracking.memory_tracker import MemoryTracker
from tests.pipeline_run import STEP_RUNS as STEPS, fake_family, run_config, ticking_clock  # noqa: F401
from tests.steps.fake_models import SPLIT, biased_dataset


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
    return run_config(tmp_path)


def run_once(cfg, release=True, tracker=None, **options):
    return run(cfg, tracker or MemoryTracker(), release=release, data=biased_dataset(kind="imbalance"),
               split=SPLIT, **options)


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
    assert (release / "model" / "manifest.json").is_file()               # the saved model folder, whole
    assert manifest["runs_on"] == "server" and manifest["model_id"] == "parkinsons-risk-v1"
    metrics = json.loads((release / "metrics.json").read_text(encoding="utf-8"))
    assert 0.5 < metrics["cv/state_now/off_auc"] <= 1
    stats = json.loads((release / "stats.json").read_text(encoding="utf-8"))
    assert stats["participants"] == len(SPLIT.development) and "move_mean" in stats["features"]
    # The drift reference: the chosen model's P(OFF) bands on its out-of-fold hours.
    reference = stats["prediction_reference"]
    assert reference["output"] == "p_off" and len(reference["counts"]) == 10 and len(reference["bands"]) == 11
    assert sum(reference["counts"]) == reference["hours"] > 0


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


def test_the_run_is_tagged_with_its_candidates_and_reason(cfg):
    tracker = MemoryTracker()
    run_once(cfg, release=False, tracker=tracker, reason="drift: move_mean")
    tags = tracker.find("model_pipeline").tags
    assert tags["candidates"] == ",".join(c["name"] for c in cfg["models"]["candidates"])
    assert tags["reason"] == "drift: move_mean" and tags["tuned_this_run"] == "false"


def test_with_tune_the_run_tunes_first_and_trains_with_tuned_settings(cfg, monkeypatch):
    tuned_with = []
    monkeypatch.setattr(TuneModels, "run",
                        lambda self, data, split: tuned_with.append(self.cfg["train"]["use_tuned"]) or {})
    cfg["train"]["use_tuned"] = False
    cfg["tune"]["candidates"] = {}                                     # the fake family has nothing to tune
    tracker = MemoryTracker()
    run_once(cfg, release=False, tracker=tracker, tune=True)
    assert tuned_with == [True]                                        # train reads what tune writes
    pipeline = tracker.find("model_pipeline")
    assert pipeline.tags["tuned_this_run"] == "true"
    assert cfg["train"]["use_tuned"] is False                          # the caller's config is unchanged


def test_a_model_no_better_than_production_is_not_exported_or_released(cfg):
    run_once(cfg)
    tracker = MemoryTracker()
    with pytest.raises(GateError, match="production gate failed"):
        run_once(cfg, tracker=tracker)
    assert versions(family_dir(cfg)) == [1] and active_version(family_dir(cfg)) == 1
    assert tracker.find("model_pipeline").status == RunStatus.FAILED
    with pytest.raises(Exception, match="expected one run named 'export'"):
        tracker.find("export")


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


def test_main_takes_the_candidates_to_train(main_with):
    assert main_with(["--no-mlflow", "--no-register", "--candidates", "fake_hourly"]) == 0


def test_main_refuses_a_candidate_the_config_doesnt_have(main_with):
    with pytest.raises(SystemExit):                                    # argparse's error, exit code 2
        main_with(["--no-mlflow", "--candidates", "nope"])


def test_a_crash_is_alerted_with_its_stack_trace_and_re_raised(main_with, alerts, monkeypatch):
    def broken(self, candidates):
        raise KeyError("missing column")
    monkeypatch.setattr(EvaluateModel, "run", broken)
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
