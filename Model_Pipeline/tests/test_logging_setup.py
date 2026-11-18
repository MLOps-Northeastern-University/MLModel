import io
import json
import logging

import pytest

from model_pipeline.utils import logging_setup
from model_pipeline.utils.logging_setup import configure_logging, get_logger, log_context, log_run_start, log_step


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("MODEL_LOG_LEVEL", "MODEL_LOG_FORMAT", "MODEL_LOG_FILE"):
        monkeypatch.delenv(var, raising=False)
    yield
    configure_logging({"logging": {"file": False}})


def capture(fmt="json", **settings):
    """Configure logging into a buffer, without a file; return the buffer."""
    buffer = io.StringIO()
    configure_logging({"logging": {"format": fmt, "file": False, **settings}}, stream=buffer)
    return buffer


def entries(buffer):
    return [json.loads(line) for line in buffer.getvalue().splitlines() if line]


def test_json_line_has_cloud_logging_fields():
    out = capture()
    get_logger("model_pipeline.steps.train").info("fold %d done", 2)
    (entry,) = entries(out)
    assert entry["severity"] == "INFO"
    assert entry["message"] == "fold 2 done"
    assert entry["logger"] == "model.steps.train"
    assert entry["timestamp"].endswith("+00:00")


def test_context_and_extra_fields():
    out = capture()
    log = get_logger("train")
    with log_context(run_id="abc", step="train"):
        with log_context(fold=3):
            log.info("fold done", extra={"pr_auc": 0.71})
        log.info("after fold")
    log.info("outside")
    inner, after, outside = entries(out)
    assert (inner["run_id"], inner["step"], inner["fold"], inner["pr_auc"]) == ("abc", "train", 3, 0.71)
    assert "fold" not in after and after["run_id"] == "abc"
    assert "run_id" not in outside


def test_exception_has_stack_trace():
    out = capture()
    try:
        raise ValueError("boom")
    except ValueError:
        get_logger("train").exception("step failed")
    (entry,) = entries(out)
    assert entry["severity"] == "ERROR" and "ValueError: boom" in entry["stack_trace"]


def test_third_party_logs_use_the_same_format():
    out = capture()
    logging.getLogger("mlflow.tracking").warning("server slow")
    logging.getLogger("mlflow.tracking").info("chatty detail")      # below third_party_level
    (entry,) = entries(out)
    assert entry["logger"] == "mlflow.tracking" and entry["severity"] == "WARNING"


def test_environment_overrides_config(monkeypatch):
    monkeypatch.setenv("MODEL_LOG_FORMAT", "json")
    monkeypatch.setenv("MODEL_LOG_LEVEL", "WARNING")
    out = capture("console", level="DEBUG")
    log = get_logger("train")
    log.info("hidden")
    log.warning("shown")
    (entry,) = entries(out)
    assert entry["message"] == "shown"


def test_reconfiguring_does_not_duplicate_lines():
    capture()
    out = capture()
    get_logger("train").info("once")
    assert len(entries(out)) == 1


def test_console_format_appends_fields():
    out = capture("console")
    with log_context(run_id="abc"):
        get_logger("train").info("hello")
    assert "| INFO     | model.train | hello  run_id=abc" in out.getvalue()


def test_log_file_written(tmp_path):
    path = tmp_path / "logs" / "run.log"
    configure_logging({"logging": {"file": True}}, log_file=path, stream=io.StringIO())
    get_logger("train").info("to file")
    for handler in logging_setup._handlers:
        handler.flush()
    assert "to file" in path.read_text(encoding="utf-8")


def test_bad_format_rejected():
    with pytest.raises(ValueError, match="console or json"):
        capture("xml")


def test_log_step_records_start_finish_and_duration():
    out = capture()
    log = get_logger("train")
    with log_step(log, "train", model="logistic"):
        log.info("inside")
    started, inside, finished = entries(out)
    assert started["event"] == "step_started" and started["model"] == "logistic"
    assert inside["step"] == "train"
    assert finished["event"] == "step_finished" and finished["duration_s"] >= 0


def test_log_step_records_failure_and_reraises():
    out = capture()
    log = get_logger("train")
    with pytest.raises(RuntimeError):
        with log_step(log, "train"):
            raise RuntimeError("no data")
    started, failed = entries(out)
    assert failed["event"] == "step_failed" and failed["severity"] == "ERROR"
    assert "RuntimeError: no data" in failed["stack_trace"] and "duration_s" in failed


def test_log_run_start_records_reproducibility_fields():
    out = capture()
    cfg = {"data": {"random_seed": 42}, "resolved_paths": {"x": "machine-specific"}}
    log_run_start(get_logger("run"), cfg, packages=("numpy",))
    other = {"data": {"random_seed": 42}, "resolved_paths": {"x": "elsewhere"}}
    log_run_start(get_logger("run"), other, packages=())
    entry, again = entries(out)
    assert entry["event"] == "run_started"
    assert set(entry) >= {"git_commit", "git_dirty", "config_sha256", "python", "packages", "host"}
    assert "numpy" in entry["packages"]
    assert again["config_sha256"] == entry["config_sha256"]     # paths don't change the hash


def test_without_git_the_images_commit_is_used(monkeypatch):
    monkeypatch.setattr(logging_setup, "_git", lambda *args: None)        # a container: no git
    monkeypatch.setenv("GIT_COMMIT", "abc123")
    env = logging_setup.environment(packages=())
    assert env["git_commit"] == "abc123" and env["git_dirty"] is False
