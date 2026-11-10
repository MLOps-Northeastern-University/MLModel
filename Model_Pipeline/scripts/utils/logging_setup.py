"""Logging for the model pipeline: readable text locally, JSON lines in the cloud.

Built on Python's logging module, so MLflow, scikit-learn and other libraries
log through the same handlers and come out in the same format.

    logger = get_logger(__name__)            # at module level, as before
    configure_logging(cfg)                   # once, at startup
    with log_context(run_id=run_id):
        log_run_start(logger, cfg)           # git commit, config hash, versions, host
        with log_step(logger, "train", model="logistic"):   # start, end, duration, failure
            logger.info("fold done", extra={"fold": 2, "pr_auc": 0.71})

What every step should log, so a run can be followed and debugged from its logs:
    - inputs: data version, row and participant counts, shapes
    - decisions: chosen settings, skipped items and why (WARNING)
    - results: metrics per fold and split, with the names MLflow uses
    - outputs: what was written and where
    - failures: logger.exception, which records the stack trace

Formats:
    console  time | level | logger | message  key=value ...   (local runs, CI)
    json     one JSON object per line on stdout: timestamp, severity, message,
             logger, then the context and extra fields. Cloud Logging (GCP) and
             CloudWatch (AWS) parse these into searchable fields; severity and
             message are the names Cloud Logging recognises.

Settings come from cfg["logging"] and can be overridden by environment
variables, so a deployment sets them without editing the config:
    MODEL_LOG_LEVEL    DEBUG, INFO, WARNING, ...
    MODEL_LOG_FORMAT   console or json (set json in the cloud)
    MODEL_LOG_FILE     0 to turn off the rotating log file (containers lose it anyway)

Never log patient data: no raw sensor values, diary answers, names or
contact details. Participant IDs, counts, shapes and metrics are fine.
"""

import contextlib
import contextvars
import datetime as dt
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from scripts.utils.config import PROJECT_ROOT

ROOT = "model"
DEFAULTS = {"level": "INFO", "format": "console", "third_party_level": "WARNING", "file": True}
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "model_pipeline.log"

# Attributes every LogRecord has; anything else on a record is an extra field.
_STANDARD = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}
_context = contextvars.ContextVar("log_context", default={})
_handlers = []


def get_logger(name):
    """Return the pipeline's logger for a module.

    Safe to call at import time: until configure_logging runs, the defaults
    (INFO, console format, with the log file) apply.
    """
    if not _handlers:
        configure_logging()
    short = name.removeprefix("scripts.")
    return logging.getLogger(f"{ROOT}.{short}")


@contextlib.contextmanager
def log_context(**fields):
    """Add fields such as run_id, step, model or fold to every line logged inside."""
    token = _context.set({**_context.get(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


def configure_logging(cfg=None, log_file=None):
    """Set up the handlers once; calling again replaces them (no duplicate lines).

    Args:
        cfg: The pipeline config. Uses its "logging" section and, for the file,
            resolved_paths["log_file"].
        log_file: Overrides the log file path.
    """
    settings = {**DEFAULTS, **((cfg or {}).get("logging") or {})}
    level = os.environ.get("MODEL_LOG_LEVEL", settings["level"]).upper()
    fmt = os.environ.get("MODEL_LOG_FORMAT", settings["format"]).lower()
    if fmt not in ("console", "json"):
        raise ValueError(f"logging format must be console or json, not {fmt!r}")
    use_file = os.environ.get("MODEL_LOG_FILE", "1" if settings["file"] else "0") != "0"

    root = logging.getLogger()
    for handler in _handlers:
        root.removeHandler(handler)
        handler.close()
    _handlers.clear()

    formatter = JsonFormatter() if fmt == "json" else ConsoleFormatter()
    stream = _StdoutHandler()
    _handlers.append(stream)
    file_error = None
    if use_file:
        path = Path(log_file or (cfg or {}).get("resolved_paths", {}).get("log_file") or DEFAULT_LOG_FILE)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _handlers.append(RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"))
        except OSError as exc:  # a read-only file system shouldn't stop the pipeline
            file_error = (path, exc)

    context_filter = _ContextFilter()
    for handler in _handlers:
        handler.setFormatter(formatter)
        handler.addFilter(context_filter)
        root.addHandler(handler)

    # Libraries log through the root logger at their own level; ours at ours.
    root.setLevel(settings["third_party_level"].upper())
    pipeline = logging.getLogger(ROOT)
    pipeline.setLevel(level)
    pipeline.propagate = True
    if file_error:
        pipeline.warning("Could not create log file %s: %s", *file_error)


@contextlib.contextmanager
def log_step(logger, step, **fields):
    """Log a step's start, its end or failure, and how long it took.

    Inside, every line carries step (and any fields given). A failure is logged
    with its stack trace and duration, then re-raised for the caller to handle.

        with log_step(logger, "train", model="logistic"):
            ...
    """
    with log_context(step=step, **fields):
        start = time.perf_counter()
        logger.info("%s started", step, extra={"event": "step_started"})
        try:
            yield
        except BaseException:
            logger.exception("%s failed", step, extra={
                "event": "step_failed", "duration_s": round(time.perf_counter() - start, 3)})
            raise
        logger.info("%s finished", step, extra={
            "event": "step_finished", "duration_s": round(time.perf_counter() - start, 3)})


def log_run_start(logger, cfg, packages=("numpy", "pandas", "scikit-learn", "mlflow")):
    """Log what's needed to reproduce a run: code version, config, environment.

    Fields: git_commit (and whether the tree had uncommitted changes),
    config_sha256 (of the config as loaded, paths excluded), python, the
    versions of the given packages that are installed, and host.
    """
    settings = {k: v for k, v in (cfg or {}).items() if k != "resolved_paths"}
    config_hash = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    logger.info("run started", extra={
        "event": "run_started",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "config_sha256": config_hash[:16],
        "python": platform.python_version(),
        "packages": versions,
        "host": platform.node(),
    })


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


class _StdoutHandler(logging.StreamHandler):
    """Writes to whatever sys.stdout is when the line is written, not when set up."""

    def __init__(self):
        super().__init__(sys.stdout)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        pass


class _ContextFilter(logging.Filter):
    """Copy the current log_context fields onto each record."""

    def filter(self, record):
        for key, value in _context.get().items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


def _extra_fields(record):
    return {k: v for k, v in vars(record).items() if k not in _STANDARD and not k.startswith("_")}


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with Cloud Logging's field names."""

    def format(self, record):
        entry = {
            "timestamp": dt.datetime.fromtimestamp(record.created, dt.timezone.utc).isoformat(),
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        entry.update(_extra_fields(record))
        if record.exc_info:
            entry["stack_trace"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """time | level | logger | message, then context and extra fields as key=value."""

    def __init__(self):
        super().__init__("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    def format(self, record):
        line = super().format(record)
        fields = _extra_fields(record)
        if fields:
            tail = " ".join(f"{k}={v}" for k, v in fields.items())
            first, _, rest = line.partition("\n")   # keep a traceback below the fields
            line = f"{first}  {tail}" + (f"\n{rest}" if rest else "")
        return line
