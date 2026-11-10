import json
import logging

import pytest

from scripts.utils import logging_setup
from scripts.utils.logging_setup import configure_logging, get_logger, log_context, log_run_start, log_step


@pytest.fixture
def json_logs(monkeypatch, capsys):
    """Configure JSON logging without a file; return a function giving the parsed lines."""
    for var in ("MODEL_LOG_LEVEL", "MODEL_LOG_FORMAT", "MODEL_LOG_FILE"):
        monkeypatch.delenv(var, raising=False)
    configure_logging({"logging": {"format": "json", "file": False}})
    capsys.readouterr()

    def lines():
        return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    yield lines
    configure_logging({"logging": {"file": False}})


def test_json_line_has_cloud_logging_fields(json_logs):
    get_logger("scripts.steps.train").info("fold %d done", 2)
    (entry,) = json_logs()
    assert entry["severity"] == "INFO"
    assert entry["message"] == "fold 2 done"
    assert entry["logger"] == "model.steps.train"
    assert entry["timestamp"].endswith("+00:00")


def test_context_and_extra_fields(json_logs):
    log = get_logger("train")
    with log_context(run_id="abc", step="train"):
        with log_context(fold=3):
            log.info("fold done", extra={"pr_auc": 0.71})
        log.info("after fold")
    log.info("outside")
    inner, after, outside = json_logs()
    assert (inner["run_id"], inner["step"], inner["fold"], inner["pr_auc"]) == ("abc", "train", 3, 0.71)
    assert "fold" not in after and after["run_id"] == "abc"
    assert "run_id" not in outside


def test_exception_has_stack_trace(json_logs):
    try:
        raise ValueError("boom")
    except ValueError:
        get_logger("train").exception("step failed")
    (entry,) = json_logs()
    assert entry["severity"] == "ERROR" and "ValueError: boom" in entry["stack_trace"]


def test_third_party_logs_use_the_same_format(json_logs):
    logging.getLogger("mlflow.tracking").warning("server slow")
    logging.getLogger("mlflow.tracking").info("chatty detail")      # below third_party_level
    (entry,) = json_logs()
    assert entry["logger"] == "mlflow.tracking" and entry["severity"] == "WARNING"


def test_environment_overrides_config(monkeypatch, capsys):
    monkeypatch.setenv("MODEL_LOG_FORMAT", "json")
    monkeypatch.setenv("MODEL_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("MODEL_LOG_FILE", "0")
    configure_logging({"logging": {"format": "console", "level": "DEBUG"}})
    capsys.readouterr()
    log = get_logger("train")
    log.info("hidden")
    log.warning("shown")
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 and json.loads(out[0])["message"] == "shown"
    monkeypatch.delenv("MODEL_LOG_FORMAT")
    monkeypatch.delenv("MODEL_LOG_LEVEL")
    configure_logging({"logging": {"file": False}})


def test_reconfiguring_does_not_duplicate_lines(json_logs, capsys):
    configure_logging({"logging": {"format": "json", "file": False}})
    configure_logging({"logging": {"format": "json", "file": False}})
    get_logger("train").info("once")
    assert len(json_logs()) == 1


def test_console_format_appends_fields(monkeypatch, capsys):
    monkeypatch.delenv("MODEL_LOG_FORMAT", raising=False)
    configure_logging({"logging": {"format": "console", "file": False}})
    capsys.readouterr()
    with log_context(run_id="abc"):
        get_logger("train").info("hello")
    line = capsys.readouterr().out.strip()
    assert "| INFO     | model.train | hello  run_id=abc" in line


def test_log_file_written(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_LOG_FILE", raising=False)
    path = tmp_path / "logs" / "run.log"
    configure_logging({"logging": {"file": True}}, log_file=path)
    get_logger("train").info("to file")
    for handler in logging_setup._handlers:
        handler.flush()
    assert "to file" in path.read_text(encoding="utf-8")
    configure_logging({"logging": {"file": False}})


def test_bad_format_rejected(monkeypatch):
    monkeypatch.delenv("MODEL_LOG_FORMAT", raising=False)
    with pytest.raises(ValueError, match="console or json"):
        configure_logging({"logging": {"format": "xml"}})


def test_log_step_records_start_finish_and_duration(json_logs):
    log = get_logger("train")
    with log_step(log, "train", model="logistic"):
        log.info("inside")
    started, inside, finished = json_logs()
    assert started["event"] == "step_started" and started["model"] == "logistic"
    assert inside["step"] == "train"
    assert finished["event"] == "step_finished" and finished["duration_s"] >= 0


def test_log_step_records_failure_and_reraises(json_logs):
    log = get_logger("train")
    with pytest.raises(RuntimeError):
        with log_step(log, "train"):
            raise RuntimeError("no data")
    started, failed = json_logs()
    assert failed["event"] == "step_failed" and failed["severity"] == "ERROR"
    assert "RuntimeError: no data" in failed["stack_trace"] and "duration_s" in failed


def test_log_run_start_records_reproducibility_fields(json_logs):
    cfg = {"data": {"random_seed": 42}, "resolved_paths": {"x": "machine-specific"}}
    log_run_start(get_logger("run"), cfg, packages=("numpy",))
    (entry,) = json_logs()
    assert entry["event"] == "run_started"
    assert set(entry) >= {"git_commit", "git_dirty", "config_sha256", "python", "packages", "host"}
    assert "numpy" in entry["packages"]
    other = {"data": {"random_seed": 42}, "resolved_paths": {"x": "elsewhere"}}
    log_run_start(get_logger("run"), other, packages=())
    (again,) = json_logs()
    assert again["config_sha256"] == entry["config_sha256"]     # paths don't change the hash
