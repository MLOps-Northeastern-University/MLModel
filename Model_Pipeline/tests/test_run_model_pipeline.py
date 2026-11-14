import pytest

from scripts import run_model_pipeline
from scripts.core.errors import GateError
from scripts.core.tracker import RunStatus
from scripts.run_model_pipeline import run
from scripts.steps.register import active_version, versions
from scripts.tracking.memory_tracker import MemoryTracker
from tests.helpers import make_config


def family_dir(cfg):
    return cfg["resolved_paths"]["registry_dir"] / cfg["export"]["family"]


def test_first_run_releases_v1(tmp_path):
    cfg = make_config(tmp_path)
    tracker = MemoryTracker()
    assert run(cfg, tracker) == 1
    assert active_version(family_dir(cfg)) == 1
    pipeline = tracker.find("model_pipeline")
    assert pipeline.status == RunStatus.FINISHED
    assert [r.name for r in tracker.children(pipeline.run_id)] == [
        "select_model", "compare_production", "export", "register"]


def test_without_release_the_gates_run_but_nothing_is_registered(tmp_path):
    cfg = make_config(tmp_path)
    tracker = MemoryTracker()
    assert run(cfg, tracker, release=False) is None
    assert versions(family_dir(cfg)) == []
    assert (cfg["resolved_paths"]["artifacts_dir"] / "bundle" / "manifest.json").is_file()
    pipeline = tracker.find("model_pipeline")
    assert [r.name for r in tracker.children(pipeline.run_id)] == ["select_model", "compare_production", "export"]


def test_a_model_no_better_than_production_is_not_exported_or_released(tmp_path):
    cfg = make_config(tmp_path)
    run(cfg, MemoryTracker())
    tracker = MemoryTracker()
    with pytest.raises(GateError, match="production gate failed"):
        run(cfg, tracker)
    assert versions(family_dir(cfg)) == [1] and active_version(family_dir(cfg)) == 1
    assert tracker.find("model_pipeline").status == RunStatus.FAILED
    with pytest.raises(Exception, match="expected one run named 'export'"):
        tracker.find("export")


@pytest.fixture
def alerts(monkeypatch):
    """Records alerts instead of sending them, whatever the machine's .env holds."""
    sent = []
    monkeypatch.setattr(run_model_pipeline, "notify", lambda subject, body, cfg=None: sent.append((subject, body)))
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    return sent


@pytest.fixture
def main_with(tmp_path, monkeypatch):
    """main(), reading a config whose outputs are under tmp_path."""
    cfg = make_config(tmp_path)
    monkeypatch.setattr(run_model_pipeline, "load_config", lambda path=None: cfg)
    return run_model_pipeline.main


def test_main_exits_0_on_release_and_1_on_a_failed_gate(main_with, alerts):
    assert main_with(["--no-mlflow", "--no-register"]) == 0
    assert main_with(["--no-mlflow"]) == 0
    assert main_with(["--no-mlflow"]) == 1


def test_a_check_only_run_sends_no_alert(main_with, alerts):
    main_with(["--no-mlflow", "--no-register"])
    assert alerts == []


def test_a_release_and_a_failed_gate_are_alerted(main_with, alerts):
    main_with(["--no-mlflow"])
    main_with(["--no-mlflow"])
    (released, _), (failed, body) = alerts
    assert released == "[MedNOVA model] parkinsons-risk v1 released"
    assert failed == "[MedNOVA model] The production gate failed"
    assert "cv/state_now/off_auc: 0.500 vs 0.500" in body and "a local run on" in body


def test_a_crash_is_alerted_with_its_stack_trace_and_re_raised(main_with, alerts, monkeypatch):
    def broken(cfg, dataset):
        raise KeyError("missing column")
    monkeypatch.setattr(run_model_pipeline, "stand_in_train", broken)
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
