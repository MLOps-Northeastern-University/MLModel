import pytest

from model_pipeline.core.errors import GateError
from model_pipeline.core.tracker import RunStatus
from model_pipeline.steps.compare_production import CompareProduction, failed_rules, passes
from model_pipeline.tracking.memory_tracker import MemoryTracker
from model_pipeline.utils.config import load_config


def test_must_improve_needs_a_higher_score():
    assert passes(0.72, 0.71, must_improve=True)
    assert not passes(0.71, 0.71, must_improve=True)


def test_hold_accepts_an_equal_score():
    assert passes(0.71, 0.71, must_improve=False)
    assert not passes(0.70, 0.71, must_improve=False)


RULES = [
    {"metric": "cv/state_now/off_auc", "must_improve": True},
    {"metric": "cv/tremor_now/auc", "must_improve": False},
]
PRODUCTION = {"cv/state_now/off_auc": 0.71, "cv/tremor_now/auc": 0.65}


def test_lists_every_failed_rule():
    new = {"cv/state_now/off_auc": 0.70, "cv/tremor_now/auc": 0.60}
    assert failed_rules(RULES, new, PRODUCTION) == [
        "cv/state_now/off_auc: 0.700 vs 0.710",
        "cv/tremor_now/auc: 0.600 vs 0.650",
    ]


def make_gate():
    tracker = MemoryTracker()
    gate = CompareProduction({"release_gate": {"rules": RULES}}, tracker)
    return gate, tracker


def test_a_better_model_passes():
    gate, tracker = make_gate()
    gate.run({"cv/state_now/off_auc": 0.74, "cv/tremor_now/auc": 0.65}, PRODUCTION)
    assert tracker.find("compare_production").tags["release_gate"] == "passed"


def test_a_worse_model_stops_the_run():
    gate, tracker = make_gate()
    with pytest.raises(GateError, match="cv/state_now/off_auc"):
        gate.run({"cv/state_now/off_auc": 0.70, "cv/tremor_now/auc": 0.65}, PRODUCTION)
    assert tracker.find("compare_production").tags["release_gate"] == "failed"


def test_the_first_model_goes_ahead():
    gate, tracker = make_gate()
    gate.run({"cv/state_now/off_auc": 0.50, "cv/tremor_now/auc": 0.50}, None)
    assert tracker.find("compare_production").tags["release_gate"] == "first_model"


def test_no_failures_when_every_rule_passes():
    new = {"cv/state_now/off_auc": 0.72, "cv/tremor_now/auc": 0.66}
    assert failed_rules(RULES, new, PRODUCTION) == []


def test_only_the_failed_rule_is_listed():
    new = {"cv/state_now/off_auc": 0.75, "cv/tremor_now/auc": 0.64}
    assert failed_rules(RULES, new, PRODUCTION) == ["cv/tremor_now/auc: 0.640 vs 0.650"]


def test_better_main_metric_with_others_held_passes():
    gate, tracker = make_gate()
    gate.run({"cv/state_now/off_auc": 0.72, "cv/tremor_now/auc": 0.65}, PRODUCTION)
    assert tracker.find("compare_production").tags["release_gate"] == "passed"


def test_a_tie_on_the_main_metric_fails():
    gate, _ = make_gate()
    with pytest.raises(GateError):
        gate.run({"cv/state_now/off_auc": 0.71, "cv/tremor_now/auc": 0.70}, PRODUCTION)


def test_a_failed_gate_marks_the_run_failed():
    gate, tracker = make_gate()
    with pytest.raises(GateError):
        gate.run({"cv/state_now/off_auc": 0.60, "cv/tremor_now/auc": 0.60}, PRODUCTION)
    assert tracker.find("compare_production").status == RunStatus.FAILED


def test_the_run_is_tagged_with_its_owner():
    gate, tracker = make_gate()
    gate.run({"cv/state_now/off_auc": 0.74, "cv/tremor_now/auc": 0.65}, PRODUCTION)
    assert tracker.find("compare_production").tags["owner"] == "P4"


def test_a_missing_metric_stops_the_run():
    gate, _ = make_gate()
    with pytest.raises(GateError, match="cv/tremor_now/auc"):
        gate.run({"cv/state_now/off_auc": 0.74}, PRODUCTION)


def test_reads_the_rules_from_the_real_config():
    gate = CompareProduction(load_config(), MemoryTracker())
    assert len(gate.rules) == 6
    assert gate.rules[0] == {"metric": "cv/state_now/off_auc", "must_improve": True}
    assert all(rule["must_improve"] is False for rule in gate.rules[1:])


def test_for_an_error_improving_means_going_down():
    rules = [{"metric": "cv/state_now/score_mae", "must_improve": True}]
    assert failed_rules(rules, {"cv/state_now/score_mae": 0.7}, {"cv/state_now/score_mae": 0.8}) == []
    assert failed_rules(rules, {"cv/state_now/score_mae": 0.9}, {"cv/state_now/score_mae": 0.8}) != []


def test_a_metric_missing_in_both_models_is_skipped():
    rules = RULES + [{"metric": "cv/state_now/dyskinetic_auc", "must_improve": False}]
    new = {"cv/state_now/off_auc": 0.74, "cv/tremor_now/auc": 0.65}
    assert failed_rules(rules, new, PRODUCTION) == []


def test_an_empty_nan_score_counts_as_missing():
    rules = [{"metric": "cv/state_now/dyskinetic_auc", "must_improve": False}]
    nan = float("nan")
    assert failed_rules(rules, {"cv/state_now/dyskinetic_auc": nan}, {"cv/state_now/dyskinetic_auc": nan}) == []
    assert failed_rules(rules, {"cv/state_now/dyskinetic_auc": nan}, {"cv/state_now/dyskinetic_auc": 0.6}) == [
        "cv/state_now/dyskinetic_auc: missing"]


def test_a_metric_the_active_model_lacks_is_skipped():
    rules = [{"metric": "cv/state_now/dyskinetic_auc", "must_improve": False}]
    assert failed_rules(rules, {"cv/state_now/dyskinetic_auc": 0.6}, {}) == []


def test_reruns_pass_when_dyskinetic_auc_is_missing_in_both():
    rules = RULES + [{"metric": "cv/state_now/dyskinetic_auc", "must_improve": True}]
    gate = CompareProduction({"release_gate": {"rules": rules}}, MemoryTracker())
    gate.run({"cv/state_now/off_auc": 0.74, "cv/tremor_now/auc": 0.65}, PRODUCTION)
