import threading

import numpy as np
import pytest

from scripts.core.errors import TrackingError
from scripts.core.tracker import RunStatus, Tracker, flatten, metric_name, validate_metric_name


class StubTracker(Tracker):
    """Records every hook call, to test the rules Tracker applies."""

    def __init__(self):
        super().__init__()
        self.calls = []
        self._next = 0

    def _start_run(self, name, parent_id, tags):
        self._next += 1
        run_id = f"r{self._next}"
        self.calls.append(("start", run_id, name, parent_id, tags))
        return run_id

    def _end_run(self, run_id, status):
        self.calls.append(("end", run_id, status))

    def _log_params(self, run_id, params):
        self.calls.append(("params", run_id, params))

    def _log_metrics(self, run_id, metrics, step):
        self.calls.append(("metrics", run_id, metrics, step))

    def _log_artifact(self, run_id, path, folder):
        self.calls.append(("artifact", run_id, path.name, folder))

    def _set_tags(self, run_id, tags):
        self.calls.append(("tags", run_id, tags))


@pytest.fixture
def tracker():
    return StubTracker()


def ends(tracker):
    return {c[1]: c[2] for c in tracker.calls if c[0] == "end"}


# ---- metric names ----------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "val/state_now/off_auc", "test/tremor_now/brier", "cv/state_next_hour/recall", "run/duration_s"])
def test_valid_metric_names(name):
    validate_metric_name(name)


@pytest.mark.parametrize("name", [
    "pr_auc", "val/state_now", "validation/state_now/off_auc", "val/state/off_auc", "cv/all/loss",
    "val/state_now/PR-AUC", "val/state_now/off_auc/x", "run/", "", None])
def test_invalid_metric_names(name):
    with pytest.raises(TrackingError):
        validate_metric_name(name)


def test_metric_name_builds_and_checks():
    assert metric_name("val", "state_now", "off_auc") == "val/state_now/off_auc"
    with pytest.raises(TrackingError):
        metric_name("dev", "state_now", "off_auc")


# ---- runs ------------------------------------------------------------------------------

def test_runs_nest_and_finish(tracker):
    with tracker.run("pipeline", tags={"owner": "P2"}) as outer:
        with tracker.run("fold-1") as inner:
            assert tracker.active_run_id == inner
        assert tracker.active_run_id == outer
    assert tracker.active_run_id is None
    starts = [c for c in tracker.calls if c[0] == "start"]
    assert starts[0][2:] == ("pipeline", None, {"owner": "P2"})
    assert starts[1][2:4] == ("fold-1", outer)
    assert ends(tracker) == {outer: RunStatus.FINISHED, inner: RunStatus.FINISHED}


def test_failure_marks_failed_and_reraises(tracker):
    with pytest.raises(ValueError):
        with tracker.run("pipeline") as outer:
            with tracker.run("fold-1") as inner:
                raise ValueError("bad fold")
    assert ends(tracker) == {outer: RunStatus.FAILED, inner: RunStatus.FAILED}
    assert tracker.active_run_id is None


def test_interrupt_marks_killed(tracker):
    with pytest.raises(KeyboardInterrupt):
        with tracker.run("pipeline") as run_id:
            raise KeyboardInterrupt
    assert ends(tracker) == {run_id: RunStatus.KILLED}


def test_run_needs_a_name(tracker):
    with pytest.raises(TrackingError):
        with tracker.run(""):
            pass


def test_threads_have_their_own_runs(tracker):
    seen = {}

    def trial(n):
        with tracker.run(f"trial-{n}") as run_id:
            seen[n] = tracker.active_run_id == run_id

    with tracker.run("tune") as parent:
        threads = [threading.Thread(target=trial, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert tracker.active_run_id == parent
    assert all(seen.values()) and len(seen) == 4


# ---- logging into a run ----------------------------------------------------------------

def test_nothing_logs_outside_a_run(tracker):
    for call in (lambda: tracker.log_params({"a": 1}),
                 lambda: tracker.log_metric("val/state_now/off_auc", 0.5),
                 lambda: tracker.set_tags({"a": "b"})):
        with pytest.raises(TrackingError, match="no active run"):
            call()


def test_params_are_flattened(tracker):
    with tracker.run("train") as run_id:
        tracker.log_params({"model": {"family": "logistic", "C": 0.1}, "seed": 42})
    assert ("params", run_id, {"model.family": "logistic", "model.C": 0.1, "seed": 42}) in tracker.calls


def test_flatten():
    assert flatten({"a": {"b": {"c": 1}}, "d": 2}) == {"a.b.c": 1, "d": 2}


def test_metrics_accept_numpy_and_record_floats(tracker):
    with tracker.run("train") as run_id:
        tracker.log_metrics({"val/state_now/off_auc": np.float32(0.5), "val/state_now/n": np.int64(3)},
                            step=np.int64(2))
    assert ("metrics", run_id, {"val/state_now/off_auc": 0.5, "val/state_now/n": 3.0}, 2) in tracker.calls


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "0.5", True, None])
def test_bad_metric_values_rejected(tracker, value):
    with tracker.run("train"):
        with pytest.raises(TrackingError):
            tracker.log_metric("val/state_now/off_auc", value)


def test_one_bad_metric_records_none(tracker):
    with tracker.run("train"):
        with pytest.raises(TrackingError):
            tracker.log_metrics({"val/state_now/off_auc": 0.5, "val/state_now/recall": float("nan")})
    assert not [c for c in tracker.calls if c[0] == "metrics"]


@pytest.mark.parametrize("step", [-1, 1.5, True])
def test_bad_steps_rejected(tracker, step):
    with tracker.run("train"):
        with pytest.raises(TrackingError, match="step"):
            tracker.log_metric("val/state_now/off_auc", 0.5, step=step)


def test_artifacts_must_exist(tracker, tmp_path):
    report = tmp_path / "report.csv"
    report.write_text("x\n")
    with tracker.run("evaluate") as run_id:
        tracker.log_artifact(report, folder="reports")
        with pytest.raises(TrackingError, match="doesn't exist"):
            tracker.log_artifact(tmp_path / "missing.csv")
    assert ("artifact", run_id, "report.csv", "reports") in tracker.calls


def test_tags_become_strings(tracker):
    with tracker.run("train") as run_id:
        tracker.set_tags({"feature_spec_version": 3})
    assert ("tags", run_id, {"feature_spec_version": "3"}) in tracker.calls


def test_a_failing_run_end_doesnt_hide_the_error_that_ended_the_run():
    from scripts.tracking.memory_tracker import MemoryTracker

    class Flaky(MemoryTracker):
        def _end_run(self, run_id, status):
            raise ConnectionError("server down")

    tracker = Flaky()
    with pytest.raises(ValueError, match="the real error"):
        with tracker.run("train"):
            raise ValueError("the real error")
    with pytest.raises(ConnectionError):                 # a clean block whose run can't be closed says so
        with tracker.run("train"):
            pass
