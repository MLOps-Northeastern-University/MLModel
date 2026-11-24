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

from model_pipeline.utils.config import PROJECT_ROOT

# Every pipeline logger is named model.<module>, under this one parent, so configure_logging can give the
# pipeline its own level, apart from the libraries' (third_party_level).
ROOT = "model"
DEFAULTS = {"level": "INFO", "format": "console", "third_party_level": "WARNING", "file": True}
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "model_pipeline.log"

# Libraries that attach their own handlers on import, which would bypass ours.
# configure_logging removes those handlers so their lines use our format too.
LIBRARIES_WITH_OWN_HANDLERS = ("mlflow",)
# MLflow's documented switch to leave logging alone; only works if set before mlflow is imported.
os.environ.setdefault("MLFLOW_CONFIGURE_LOGGING", "false")

# Attributes every LogRecord has; anything else on a record is an extra field.
_STANDARD = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}
# Fields set by log_context. A ContextVar gives each thread its own copy.
# The default dict is shared by every context, which is safe only because log_context never changes it in
# place: it sets a new, merged dict and restores the previous one on exit.
_context = contextvars.ContextVar("log_context", default={})
# The handlers we added, so configure_logging can replace exactly these.
_handlers = []


def get_logger(name):
    """Return the pipeline's logger for a module.

    Safe to call at import time: until configure_logging runs, the defaults
    (INFO, console format, with the log file) apply.
    """
    if not _handlers:
        configure_logging()
    short = name.removeprefix("model_pipeline.")
    return logging.getLogger(f"{ROOT}.{short}")


@contextlib.contextmanager
def log_context(**fields):
    """Add fields such as run_id, step, model or fold to every line logged inside."""
    token = _context.set({**_context.get(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


def configure_logging(cfg=None, log_file=None, stream=None):
    """Set up the handlers once; calling again replaces them (no duplicate lines).

    Args:
        cfg: The pipeline config. Uses its "logging" section and, for the file,
            resolved_paths["log_file"].
        log_file: Overrides the log file path.
        stream: Where console or JSON lines go. Defaults to sys.stdout.
    """
    # Defaults, then the config file, then environment variables (highest priority).
    settings = {**DEFAULTS, **((cfg or {}).get("logging") or {})}
    level = os.environ.get("MODEL_LOG_LEVEL", settings["level"]).upper()
    fmt = os.environ.get("MODEL_LOG_FORMAT", settings["format"]).lower()
    if fmt not in ("console", "json"):
        raise ValueError(f"logging format must be console or json, not {fmt!r}")
    # MODEL_LOG_FILE=0 turns the file off; any other value turns it on, even when the config turns it off.
    use_file = os.environ.get("MODEL_LOG_FILE", "1" if settings["file"] else "0") != "0"

    # Remove our previous handlers first, or every line would print twice.
    root = logging.getLogger()
    for handler in _handlers:
        root.removeHandler(handler)
        handler.close()
    _handlers.clear()

    formatter = JsonFormatter() if fmt == "json" else ConsoleFormatter()
    _handlers.append(logging.StreamHandler(stream or sys.stdout))
    file_error = None
    if use_file:
        path = Path(log_file or (cfg or {}).get("resolved_paths", {}).get("log_file") or DEFAULT_LOG_FILE)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # At most about 20 MB on disk: the file rolls over at 5 MB and the three newest old files are kept.
            _handlers.append(RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"))
        except OSError as exc:  # a read-only file system shouldn't stop the pipeline
            file_error = (path, exc)

    # The filter goes on the handlers, not on a logger: a logger's filters skip records that propagate up from
    # other loggers, so only this way do library lines (MLflow's, say) get run_id and step too.
    context_filter = _ContextFilter()
    for handler in _handlers:
        handler.setFormatter(formatter)
        handler.addFilter(context_filter)
        root.addHandler(handler)

    # Handlers sit on the root logger, so library logs (MLflow, scikit-learn) use them too.
    # Libraries are held to third_party_level; the pipeline's own "model.*" loggers to level.
    root.setLevel(settings["third_party_level"].upper())
    for name in LIBRARIES_WITH_OWN_HANDLERS:   # in case the library was imported first
        library = logging.getLogger(name)
        library.handlers.clear()
        library.setLevel(logging.NOTSET)       # follow third_party_level
        library.propagate = True
    pipeline = logging.getLogger(ROOT)
    pipeline.setLevel(level)
    pipeline.propagate = True
    # Warned only now, once the handlers are in place, so the warning is formatted like every other line.
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
        except BaseException:   # includes Ctrl+C, so an interrupted step is logged too
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
    # Leave out resolved_paths: they differ per machine, so the same config would hash differently.
    settings = {k: v for k, v in (cfg or {}).items() if k != "resolved_paths"}
    config_hash = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()
    logger.info("run started", extra={
        "event": "run_started", **environment(packages),
        "config_sha256": config_hash[:16], "host": platform.node()})


def environment(packages=("numpy", "pandas", "scikit-learn", "mlflow")):
    """The code and library versions a result came from: git commit, uncommitted changes, Python, packages.

    In a container there's no git: the image's GIT_COMMIT (set when CI builds it) names the commit,
    and an image is built from a clean checkout, so it has no uncommitted changes.
    """
    # A package that isn't installed (torch or optuna, which experiments.py asks for) is left out rather than
    # raising.
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    commit = _git("rev-parse", "HEAD")
    if commit:
        dirty = bool(_git("status", "--porcelain"))
    else:
        commit, dirty = os.environ.get("GIT_COMMIT"), False
    return {"git_commit": commit, "git_dirty": dirty, "python": platform.python_version(), "packages": versions}


def _git(*args):
    """Run a git command and return its output, or None if git isn't available (e.g. in a container)."""
    try:
        # cwd=PROJECT_ROOT, so the commit is this repository's whatever folder the command ran from; the timeout
        # keeps a slow or stuck git from holding up the start of a run.
        out = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


class _ContextFilter(logging.Filter):
    """Copy the current log_context fields onto each record."""

    def filter(self, record):
        for key, value in _context.get().items():
            if not hasattr(record, key):   # a field passed in extra= wins over the context
                setattr(record, key, value)
        return True


def _extra_fields(record):
    """The record's fields beyond the standard ones: what extra= and log_context put on it.

    Names starting with "_" are internal and left out.
    """
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
        # default=str: values JSON can't hold (paths, numpy numbers) are written as text.
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
