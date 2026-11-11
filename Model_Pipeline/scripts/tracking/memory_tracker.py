"""A tracker that keeps everything in memory, for tests.

Owner: P2.

Lets a test check what a step logged (runs, parameters, metrics, artifacts)
without MLflow installed or running.

    tracker = MemoryTracker()
    train(..., tracker=tracker)
    run = tracker.find("fold-1")
    assert run.metrics["val/off/pr_auc"] == [(None, 0.71)]
"""

import itertools
import threading
from dataclasses import dataclass, field

from scripts.core.errors import TrackingError
from scripts.core.tracker import Tracker


@dataclass
class RecordedRun:
    """Everything logged into one run."""

    run_id: str
    name: str
    parent_id: str = None
    tags: dict = field(default_factory=dict)
    status: object = None          # a RunStatus once the run has ended
    params: dict = field(default_factory=dict)
    # Each metric keeps every value logged, in order, as (step, value).
    metrics: dict = field(default_factory=dict)
    # (path, folder) for each artifact.
    artifacts: list = field(default_factory=list)

    def latest(self, name):
        """The most recent value of a metric."""
        return self.metrics[name][-1][1]


class MemoryTracker(Tracker):
    """Stores runs in a dict. Safe to use from several threads."""

    def __init__(self):
        super().__init__()
        self.runs = {}                  # run_id -> RecordedRun, in creation order
        self._ids = itertools.count(1)
        self._lock = threading.Lock()   # parallel tuning trials write at the same time

    # ---- reading back, for tests -----------------------------------------------------------

    def find(self, name):
        """The one run with this name. Raises if there are none or several."""
        found = [r for r in self.runs.values() if r.name == name]
        if len(found) != 1:
            raise TrackingError(f"expected one run named {name!r}, found {len(found)}")
        return found[0]

    def children(self, run_id):
        """The runs started directly inside run_id."""
        return [r for r in self.runs.values() if r.parent_id == run_id]

    # ---- the Tracker hooks -----------------------------------------------------------------

    def _start_run(self, name, parent_id, tags):
        with self._lock:
            run_id = f"run-{next(self._ids)}"
            self.runs[run_id] = RecordedRun(run_id, name, parent_id, dict(tags))
        return run_id

    def _end_run(self, run_id, status):
        self.runs[run_id].status = status

    def _log_params(self, run_id, params):
        with self._lock:
            self.runs[run_id].params.update(params)

    def _log_metrics(self, run_id, metrics, step):
        with self._lock:
            for name, value in metrics.items():
                self.runs[run_id].metrics.setdefault(name, []).append((step, value))

    def _log_artifact(self, run_id, path, folder):
        with self._lock:
            self.runs[run_id].artifacts.append((path, folder))

    def _set_tags(self, run_id, tags):
        with self._lock:
            self.runs[run_id].tags.update(tags)
