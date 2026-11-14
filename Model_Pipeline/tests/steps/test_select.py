import pytest

from scripts.core.errors import GateError
from scripts.steps.select import SelectModel, best, eligible
from scripts.tracking.memory_tracker import MemoryTracker
from scripts.utils.config import load_config

METRIC = "cv/state_now/off_auc"


def candidate(name, score, gates_passed=True, exportable=True):
    return {"name": name, "metrics": {METRIC: score}, "gates_passed": gates_passed, "exportable": exportable}


def make_step():
    tracker = MemoryTracker()
    return SelectModel({"selection": {"metric": METRIC}}, tracker), tracker


def test_eligible_drops_failed_gates_and_unexportable_models():
    kept = eligible([
        candidate("good", 0.70),
        candidate("failed_bias", 0.90, gates_passed=False),
        candidate("no_phone_format", 0.95, exportable=False),
    ])
    assert [c["name"] for c in kept] == ["good"]


def test_best_picks_the_highest_score():
    assert best([candidate("a", 0.70), candidate("b", 0.74), candidate("c", 0.72)], METRIC)["name"] == "b"


def test_best_keeps_the_first_on_a_tie():
    assert best([candidate("a", 0.74), candidate("b", 0.74)], METRIC)["name"] == "a"


def test_best_with_no_candidates_stops_the_run():
    with pytest.raises(GateError, match="selection gate failed"):
        best([], METRIC)


def test_run_returns_the_best_eligible_candidate():
    step, tracker = make_step()
    chosen = step.run([candidate("a", 0.70), candidate("b", 0.95, gates_passed=False), candidate("c", 0.72)])
    assert chosen["name"] == "c"
    run = tracker.find("select_model")
    assert run.tags == {"owner": "P4", "selected_model": "c"}


def test_run_with_no_eligible_candidate_stops_the_run():
    step, _ = make_step()
    with pytest.raises(GateError, match="no eligible candidates"):
        step.run([candidate("a", 0.90, exportable=False)])


def test_reads_the_metric_from_the_real_config():
    step = SelectModel(load_config(), MemoryTracker())
    assert step.metric == METRIC


def test_best_takes_the_lowest_score_for_an_error():
    error = "cv/state_now/score_mae"
    candidates = [{"name": n, "metrics": {error: v}, "gates_passed": True, "exportable": True}
                  for n, v in (("a", 0.9), ("b", 0.6), ("c", 0.7))]
    assert best(candidates, error)["name"] == "b"
