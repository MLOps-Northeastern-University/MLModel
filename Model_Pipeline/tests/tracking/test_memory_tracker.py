import contextvars
import threading

import pytest

from scripts.core.errors import TrackingError
from scripts.core.tracker import RunStatus
from scripts.tracking.memory_tracker import MemoryTracker


@pytest.fixture
def tracker():
    return MemoryTracker()


def test_records_a_nested_run(tracker, tmp_path):
    report = tmp_path / "report.csv"
    report.write_text("x\n")
    with tracker.run("train", tags={"owner": "P2"}) as parent:
        tracker.log_params({"model": {"C": 0.1}})
        with tracker.run("fold-1"):
            tracker.log_metric("val/state_now/off_auc", 0.6, step=0)
            tracker.log_metric("val/state_now/off_auc", 0.7, step=1)
            tracker.log_artifact(report, folder="reports")
            tracker.set_tags({"fold": 1})

    train, fold = tracker.find("train"), tracker.find("fold-1")
    assert train.tags == {"owner": "P2"} and train.params == {"model.C": 0.1}
    assert fold.parent_id == parent and tracker.children(parent) == [fold]
    assert fold.metrics["val/state_now/off_auc"] == [(0, 0.6), (1, 0.7)]
    assert fold.latest("val/state_now/off_auc") == 0.7
    assert fold.artifacts == [(report, "reports")] and fold.tags == {"fold": "1"}
    assert train.status == fold.status == RunStatus.FINISHED


def test_records_failure(tracker):
    with pytest.raises(ValueError):
        with tracker.run("train"):
            raise ValueError("bad data")
    assert tracker.find("train").status == RunStatus.FAILED


def test_find_needs_exactly_one(tracker):
    with pytest.raises(TrackingError, match="found 0"):
        tracker.find("missing")
    for _ in range(2):
        with tracker.run("trial"):
            pass
    with pytest.raises(TrackingError, match="found 2"):
        tracker.find("trial")


def test_parallel_trials_each_get_their_own_run(tracker):
    def trial(n):
        with tracker.run(f"trial-{n}"):
            for step in range(50):
                tracker.log_metric("val/state_now/off_auc", n / 10, step=step)

    with tracker.run("tune") as parent:
        # Each thread gets a copy of the current context, so its run nests under "tune".
        threads = [threading.Thread(target=contextvars.copy_context().run, args=(trial, n)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    trials = tracker.children(parent)
    assert len(trials) == 8
    for run in trials:
        n = int(run.name.split("-")[1])
        assert run.metrics["val/state_now/off_auc"] == [(s, n / 10) for s in range(50)]


def test_a_plain_thread_starts_outside_any_run(tracker):
    with tracker.run("tune"):
        seen = []
        t = threading.Thread(target=lambda: seen.append(tracker.active_run_id))
        t.start()
        t.join()
    assert seen == [None]
