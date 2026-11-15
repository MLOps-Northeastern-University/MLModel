import pytest

from scripts.core.errors import GateError
from scripts.steps.select import SelectModel, best, eligible
from scripts.tracking.memory_tracker import MemoryTracker
from scripts.utils.config import load_config

METRIC = "cv/state_now/off_auc"


def candidate(name, score, gates_passed=True, exportable=None):
    c = {"name": name, "metrics": {METRIC: score}, "gates_passed": gates_passed}
    if exportable is not None:
        c["exportable"] = exportable
    return c


def make_step(need_device=False):
    tracker = MemoryTracker()
    cfg = {"selection": {"metric": METRIC}, "export": {"device": {"required": need_device}}}
    return SelectModel(cfg, tracker), tracker


def test_eligible_drops_failed_gates():
    kept = eligible([candidate("good", 0.70), candidate("failed_bias", 0.90, gates_passed=False)])
    assert [c["name"] for c in kept] == ["good"]


def test_a_server_model_needs_no_exportable_field():
    assert [c["name"] for c in eligible([candidate("server_only", 0.70)])] == ["server_only"]


def test_a_candidate_without_gates_passed_is_not_eligible():
    assert eligible([{"name": "unchecked", "metrics": {METRIC: 0.9}}]) == []


def test_needing_the_device_drops_models_that_cant_be_exported():
    kept = eligible([candidate("phone_ok", 0.70, exportable=True), candidate("server_only", 0.95)],
                    need_device=True)
    assert [c["name"] for c in kept] == ["phone_ok"]


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
        step.run([candidate("a", 0.90, gates_passed=False)])


def test_run_needing_the_device_picks_an_exportable_model():
    step, _ = make_step(need_device=True)
    chosen = step.run([candidate("server_only", 0.95), candidate("phone_ok", 0.70, exportable=True)])
    assert chosen["name"] == "phone_ok"


def test_reads_the_metric_from_the_real_config():
    cfg = load_config()
    step = SelectModel(cfg, MemoryTracker())
    assert step.metric == cfg["selection"]["metric"]


def test_best_takes_the_lowest_score_for_an_error():
    error = "cv/state_now/score_mae"
    candidates = [{"name": n, "metrics": {error: v}, "gates_passed": True, "exportable": True}
                  for n, v in (("a", 0.9), ("b", 0.6), ("c", 0.7))]
    assert best(candidates, error)["name"] == "b"
