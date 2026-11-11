"""The contract for experiment tracking: runs, parameters, metrics and artifacts.

Owner: P2.

Steps and models log through a Tracker passed to them, never through MLflow
directly, so tracking can be swapped (tracking/) and tested without a server.
Runs nest: one pipeline run, then a child run per step, model part, fold or
tuning trial. Metric names follow one scheme, {split}/{target}/{metric}, and a
run always ends as finished or failed.

    with tracker.run("train", tags={"owner": "P2"}):
        tracker.log_params({"model": {"family": "logistic", "C": 0.1}})
        for fold in folds:
            with tracker.run(f"fold-{fold}"):
                tracker.log_metric(metric_name("val", "off", "pr_auc"), 0.71)

The rules live here, in Tracker, so every implementation applies them the same
way. An implementation (tracking/memory_tracker.py, tracking/mlflow_tracker.py)
only writes the six _hooks at the bottom.
"""

import contextlib
import contextvars
import enum
import math
import numbers
import re
from abc import ABC, abstractmethod
from pathlib import Path

from scripts.core.errors import TrackingError
from scripts.core.targets import TARGETS
from scripts.utils.logging_setup import get_logger, log_context

logger = get_logger(__name__)

# The splits a metric can belong to. "cv" is a summary across folds.
SPLITS = ("train", "val", "test", "cv")
# Metrics that span every target, such as a combined loss.
ALL_TARGETS = "all"
# Run-level metrics with no split or target: run/<metric>, such as run/duration_s.
RUN_NAMESPACE = "run"
_METRIC_PART = re.compile(r"^[a-z0-9_]+$")


class RunStatus(enum.Enum):
    """How a run ended. The names match MLflow's."""

    FINISHED = "FINISHED"
    FAILED = "FAILED"
    KILLED = "KILLED"


def metric_name(split, target, metric):
    """Build a metric name, {split}/{target}/{metric}, checking each part.

    Args:
        split: One of SPLITS.
        target: A target from core/targets.py, or ALL_TARGETS.
        metric: Lowercase letters, digits and underscores, such as pr_auc.

    Raises:
        TrackingError: A part isn't allowed.
    """
    name = f"{split}/{target}/{metric}"
    validate_metric_name(name)
    return name


def validate_metric_name(name):
    """Check a metric name: {split}/{target}/{metric} or run/{metric}.

    Raises:
        TrackingError: The name doesn't follow the scheme.
    """
    parts = name.split("/") if isinstance(name, str) else []
    if len(parts) == 2 and parts[0] == RUN_NAMESPACE and _METRIC_PART.match(parts[1]):
        return
    if len(parts) == 3:
        split, target, metric = parts
        if split in SPLITS and (target in TARGETS or target == ALL_TARGETS) and _METRIC_PART.match(metric):
            return
    raise TrackingError(
        f"metric name {name!r} must be {{split}}/{{target}}/{{metric}} with split in {SPLITS}, "
        f"target in {sorted(TARGETS) + [ALL_TARGETS]} and metric of a-z, 0-9 and _; "
        f"or {RUN_NAMESPACE}/{{metric}}")


def flatten(params, prefix=""):
    """Flatten nested dicts into dotted keys: {"model": {"C": 1}} -> {"model.C": 1}."""
    flat = {}
    for key, value in params.items():
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten(value, full))
        else:
            flat[full] = value
    return flat


class Tracker(ABC):
    """Records runs, parameters, metrics and artifacts.

    Not tied to one thread: each thread or async task has its own stack of
    active runs, so parallel tuning trials don't log into each other's runs.
    """

    def __init__(self):
        self._stack = contextvars.ContextVar(f"tracker_runs_{id(self)}", default=())

    @property
    def active_run_id(self):
        """The innermost active run's id, or None outside any run."""
        stack = self._stack.get()
        return stack[-1] if stack else None

    @contextlib.contextmanager
    def run(self, name, tags=None):
        """Start a run, nested in the active one if any, and end it on exit.

        The run ends FINISHED, FAILED if the block raises, or KILLED on
        KeyboardInterrupt or SystemExit. The exception is always re-raised.

        Yields:
            The run's id.
        """
        if not name or not isinstance(name, str):
            raise TrackingError("a run needs a name")
        parent = self.active_run_id
        run_id = self._start_run(name, parent, dict(tags or {}))
        token = self._stack.set(self._stack.get() + (run_id,))
        status = RunStatus.FAILED
        try:
            with log_context(tracker_run_id=run_id):
                logger.info("tracking run %s started", name, extra={
                    "event": "tracking_run_started", "tracker_parent_run_id": parent})
                try:
                    yield run_id
                    status = RunStatus.FINISHED
                except (KeyboardInterrupt, SystemExit):
                    status = RunStatus.KILLED
                    raise
                finally:
                    logger.info("tracking run %s ended %s", name, status.value, extra={
                        "event": "tracking_run_ended", "status": status.value})
        finally:
            self._stack.reset(token)
            self._end_run(run_id, status)

    def log_params(self, params):
        """Record parameters. Nested dicts are flattened to dotted keys."""
        if not isinstance(params, dict):
            raise TrackingError(f"params must be a dict, not {type(params).__name__}")
        flat = flatten(params)
        if any(not k for k in flat):
            raise TrackingError("parameter names can't be empty")
        if flat:
            self._log_params(self._require_run(), flat)

    def log_metric(self, name, value, step=None):
        """Record one metric. See log_metrics."""
        self.log_metrics({name: value}, step=step)

    def log_metrics(self, metrics, step=None):
        """Record metrics by name, checking every name and value first.

        Args:
            metrics: {name: value}. Names follow metric_name's scheme; values
                must be finite real numbers (numpy scalars are fine).
            step: Optional non-negative integer, such as an epoch.

        Raises:
            TrackingError: No active run, a bad name, value or step. Nothing is
                recorded if any metric is bad.
        """
        if step is not None and (isinstance(step, bool) or not isinstance(step, numbers.Integral) or step < 0):
            raise TrackingError(f"step must be a non-negative integer, not {step!r}")
        clean = {}
        for name, value in metrics.items():
            validate_metric_name(name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TrackingError(f"metric {name!r} must be a number, not {type(value).__name__}")
            value = float(value)
            if not math.isfinite(value):
                raise TrackingError(f"metric {name!r} is {value}; only finite values are recorded")
            clean[name] = value
        if clean:
            self._log_metrics(self._require_run(), clean, None if step is None else int(step))

    def log_artifact(self, path, folder=None):
        """Record a file or a folder, optionally under folder in the run's artifacts."""
        path = Path(path)
        if not path.exists():
            raise TrackingError(f"artifact {path} doesn't exist")
        self._log_artifact(self._require_run(), path, folder)

    def set_tags(self, tags):
        """Label the active run, such as data_version, git_commit or owner."""
        if tags:
            self._set_tags(self._require_run(), {str(k): str(v) for k, v in tags.items()})

    def _require_run(self):
        run_id = self.active_run_id
        if run_id is None:
            raise TrackingError("no active run: log inside `with tracker.run(...)`")
        return run_id

    # ---- what an implementation writes ---------------------------------------------------

    @abstractmethod
    def _start_run(self, name, parent_id, tags):
        """Create a run (a child of parent_id if not None) and return its id."""

    @abstractmethod
    def _end_run(self, run_id, status):
        """Close the run with a RunStatus."""

    @abstractmethod
    def _log_params(self, run_id, params):
        """Record already-flattened {name: value} parameters."""

    @abstractmethod
    def _log_metrics(self, run_id, metrics, step):
        """Record already-checked {name: float} metrics at step (or None)."""

    @abstractmethod
    def _log_artifact(self, run_id, path, folder):
        """Record an existing file or folder."""

    @abstractmethod
    def _set_tags(self, run_id, tags):
        """Record {str: str} tags."""
