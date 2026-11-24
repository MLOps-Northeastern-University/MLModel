"""Each step run on its own, one command at a time, as the Airflow DAG runs them: the same result as
the whole run, with every hand-off through files in artifacts/."""

import functools
import json

import pytest

from model_pipeline import pipeline_steps, run_model_pipeline, run_step
from model_pipeline.core.tracker import RunStatus
from model_pipeline.steps import bias
from model_pipeline.steps.register import active_version, versions
from model_pipeline.tracking.memory_tracker import MemoryTracker
from tests.steps.fake_models import SPLIT, biased_dataset
from tests.pipeline_run import STEP_RUNS, fake_family, run_config, ticking_clock  # noqa: F401

# Every step in pipeline_steps.STEPS but tune, in order: these runs don't tune.
ALL_STEPS = ["train", "evaluate", "bias", "explain", "select", "compare", "export", "register"]


@pytest.fixture
def cfg(tmp_path):
    return run_config(tmp_path)


@pytest.fixture(autouse=True)
def alerts(monkeypatch):
    """Records alerts instead of sending them."""
    sent = []
    record = lambda subject, body, cfg=None: sent.append(subject)   # noqa: E731
    # run_step imports notify for finish's release alert, and alert_failure uses run_model_pipeline's, so
    # both are replaced, along with the one bias imports.
    for module in (run_model_pipeline, run_step, bias):
        monkeypatch.setattr(module, "notify", record)
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    return sent


@pytest.fixture
def command(cfg, monkeypatch):
    """run_step's main, with the test config, the small data and split, and one tracker for every command
    (in a real run each command is a new process, and MLflow joins them by the run id in run.json)."""
    data = biased_dataset(kind="imbalance")
    monkeypatch.setattr(run_step, "load_config", lambda path=None: cfg)
    # start reads the split itself, for the run's tags. A real step reads the data and split from the config's
    # paths, and run_step calls it with (cfg, tracker) only, so each one is wrapped to add the small ones.
    monkeypatch.setattr(run_step, "read_split", lambda cfg: SPLIT)
    monkeypatch.setattr(run_step, "STEPS", {name: (functools.partial(function, data=data, split=SPLIT), owner)
                                            for name, (function, owner) in pipeline_steps.STEPS.items()})
    tracker = MemoryTracker()
    command = functools.partial(run_step.main, tracker_factory=lambda cfg: tracker)
    command.tracker = tracker
    return command


def run_all(command, steps=ALL_STEPS, start=()):
    """start, the steps in order until one fails (Airflow skips the rest), then finish. {step: exit code}."""
    assert command(["start", *start]) == 0
    codes = {}
    for name in steps:
        codes[name] = command([name])
        if codes[name] != 0:
            break
    failed = any(codes.values())
    command(["finish", "--status", "failed" if failed else "finished"])
    return codes


def test_the_steps_one_by_one_release_v1_under_one_mlflow_run(command, cfg, alerts):
    assert set(run_all(command).values()) == {0}
    family_dir = cfg["resolved_paths"]["registry_dir"] / cfg["export"]["family"]
    assert active_version(family_dir) == 1
    tracker = command.tracker
    pipeline = tracker.find("model_pipeline")
    assert pipeline.status == RunStatus.FINISHED
    assert [r.name for r in tracker.children(pipeline.run_id)] == STEP_RUNS + ["register"]   # as the whole run
    # run.json is how the commands share the run: start wrote the MLflow run id, register the version.
    run = json.loads((cfg["resolved_paths"]["artifacts_dir"] / "run.json").read_text(encoding="utf-8"))
    assert run["released_version"] == 1 and run["mlflow_run_id"] == pipeline.run_id
    assert alerts[-1] == "[MedNOVA model] parkinsons-risk v1 released"


def test_evaluate_saves_the_validation_gate_for_the_steps_after_it(command, cfg):
    command(["start"])
    command(["train"])
    command(["evaluate"])
    saved = json.loads((cfg["resolved_paths"]["artifacts_dir"] / "candidates.json").read_text(encoding="utf-8"))
    assert all("gates_passed" in c for c in saved)          # bias, in its own process, reads these


def test_a_failed_gate_fails_its_step_and_the_run(command, cfg, alerts):
    cfg["bias"]["max_recall_gap"] = 0.01
    codes = run_all(command)
    # bias exits 1 and run_all stops there, as Airflow skips the tasks after a failed one; finish then closes
    # the run FAILED.
    assert codes["bias"] == 1 and "[Model pipeline] Bias gate failed" in alerts
    assert command.tracker.find("model_pipeline").status == RunStatus.FAILED


def test_a_run_started_with_no_register_releases_nothing(command, cfg, alerts):
    assert set(run_all(command, start=["--no-register"]).values()) == {0}
    assert versions(cfg["resolved_paths"]["registry_dir"] / cfg["export"]["family"]) == []
    assert alerts[-1] == "[MedNOVA model] Training finished, every gate passed"


def test_a_step_before_start_is_refused(command):
    # No run.json yet, so the step has no run to log under and exits 1 without running.
    assert command(["evaluate"]) == 1


def test_start_refuses_a_candidate_the_config_doesnt_have(command):
    assert command(["start", "--candidates", "nope"]) == 2


def test_a_failed_start_leaves_no_run_for_finish_to_close(command, cfg, monkeypatch):
    run_all(command)                                            # an earlier run, finished
    earlier = command.tracker.find("model_pipeline").run_id

    # start removes the earlier run.json before it reads the split, so when the split can't be read, finish
    # finds nothing to close and the earlier run keeps its FINISHED status.
    def no_split(cfg):
        raise run_step.PipelineError("no patient split")
    monkeypatch.setattr(run_step, "read_split", no_split)
    assert command(["start"]) == 1
    assert not (cfg["resolved_paths"]["artifacts_dir"] / "run.json").exists()
    assert command(["finish", "--status", "failed"]) == 0      # nothing to close, and says so
    assert command.tracker.runs[earlier].status == RunStatus.FINISHED   # the earlier run is left alone
