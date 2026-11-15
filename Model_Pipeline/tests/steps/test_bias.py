"""The bias step: slices, the gate, reweighing, and a run from train's real outputs to the report."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.core.errors import ConfigError, GateError, SchemaError
from scripts.models import registry
from scripts.steps import bias
from scripts.steps.bias import (CheckBias, check_gaps, group_scores, reweighing_weights, settings,
                                slice_values)
from scripts.steps.train import TrainModels
from scripts.tracking.memory_tracker import MemoryTracker
from tests.steps.fake_models import SPLIT, biased_dataset, pipeline_config


@pytest.fixture(autouse=True)
def fake_family(monkeypatch):
    monkeypatch.setitem(registry.FAMILIES, "fake_hourly", "tests.steps.fake_models:HourlyLogistic")


@pytest.fixture
def alerts(monkeypatch):
    sent = []
    monkeypatch.setattr(bias, "notify", lambda subject, body, cfg=None: sent.append((subject, body)))
    return sent


def trained(tmp_path, kind="signal", seed=0, **bias_settings):
    """Train the fake family on biased data, as the pipeline does; return (cfg, data, candidates)."""
    cfg = pipeline_config(tmp_path, **bias_settings)
    data = biased_dataset(kind=kind, seed=seed)
    candidates = TrainModels(cfg, MemoryTracker()).run(data, SPLIT)
    return cfg, data, candidates


# ---- settings and slices ------------------------------------------------------------------------

def test_a_check_without_its_own_gap_uses_max_recall_gap(tmp_path):
    opts = settings(pipeline_config(tmp_path))
    assert opts["checks"][0]["max_gap"] == 0.10 and opts["checks"][1]["max_gap"] == 0.5
    assert opts["predictions"] == "cv" and opts["min_participants"] == 2


@pytest.mark.parametrize("change, message", [
    ({"checks": []}, "at least one"),
    ({"checks": [{"label": "state_later", "metric": "off_auc"}]}, "unknown label"),
    ({"predictions": "train"}, "cv or test"),
    ({"threshold": 1.5}, "threshold"),
])
def test_bad_settings_are_refused(tmp_path, change, message):
    with pytest.raises(ConfigError, match=message):
        settings(pipeline_config(tmp_path, **change))


def test_slices_come_from_the_hours_or_else_the_patients_table():
    data = biased_dataset(hours=3)
    slices = slice_values(data, ["sex", "dbs"])
    assert len(slices) == len(data.hours) and list(slices.columns[:2]) == ["participant_id", "hour_index"]
    assert set(slices["sex"]) == {"female", "male"}
    assert set(slices["dbs"]) == {"yes", "no"}                  # booleans, from the patients table
    with pytest.raises(SchemaError, match="pd_subtype"):
        slice_values(data, ["pd_subtype"])


def test_a_missing_slice_value_becomes_its_own_unknown_group():
    data = biased_dataset(hours=3)
    data.hours.loc[data.hours["participant_id"] == "P01", "sex"] = None
    assert slice_values(data, ["sex"]).loc[lambda s: s.participant_id == "P01", "sex"].eq("unknown").all()


# ---- scores and the gate ------------------------------------------------------------------------

def frame(y, p_off, people):
    """A state_now predictions frame: P(off) all on slight_akinesia, the rest on good_kinesia."""
    from scripts.core.targets import STATE_CLASSES
    out = pd.DataFrame({"participant_id": people, "y_true": y})
    for c in STATE_CLASSES:
        out[c] = 0.0
    out["slight_akinesia"] = p_off
    out["good_kinesia"] = 1 - np.asarray(p_off)
    return out


def test_question_rates_count_p_yes_at_the_threshold():
    # Two OFF hours (classes 1 and 2) and two ON hours (3); P(off) 0.9, 0.4, 0.6, 0.1.
    f = frame([1.0, 2.0, 3.0, 3.0], [0.9, 0.4, 0.6, 0.1], ["A", "A", "B", "B"])
    scores = group_scores("state_now", f, threshold=0.5)
    assert scores["off_recall"] == 0.5 and scores["off_precision"] == 0.5 and scores["off_fpr"] == 0.5
    assert scores["off_auc"] == 0.75                              # from core/metrics, the shared score


def groups(**values):
    """slice_scores-shaped groups: name=(metric value, hours, patients)."""
    return {g: {"hours": h, "participants": p, "metrics": {"off_recall": v}} for g, (v, h, p) in values.items()}


def opts(**kw):
    return {"min_rows": 20, "min_participants": 2, **kw}


def test_the_gap_is_between_the_best_and_worst_judged_group():
    slices = {"state_now": {"sex": groups(female=(0.9, 100, 5), male=(0.6, 100, 5))}}
    check = [{"label": "state_now", "metric": "off_recall", "max_gap": 0.1}]
    [r] = check_gaps(slices, check, ["sex"], opts())
    assert r["status"] == "fail" and r["gap"] == pytest.approx(0.3)
    assert (r["worst_group"], r["best_group"]) == ("male", "female")


def test_groups_with_too_few_patients_or_hours_are_not_judged():
    slices = {"state_now": {"sex": groups(female=(0.9, 100, 5), male=(0.2, 100, 1), other=(0.1, 5, 3),
                                          unknown=(0.0, 100, 9))}}
    check = [{"label": "state_now", "metric": "off_recall", "max_gap": 0.1}]
    [r] = check_gaps(slices, check, ["sex"], opts())
    assert r["status"] == "not_judged" and r["judged_groups"] == ["female"]
    assert set(r["skipped_groups"]) == {"male", "other", "unknown"}


def test_a_check_naming_a_score_the_label_lacks_is_an_error():
    slices = {"state_now": {"sex": groups(female=(0.9, 100, 5), male=(0.6, 100, 5))}}
    with pytest.raises(ConfigError, match="no score 'asleep_auc'"):
        check_gaps(slices, [{"label": "state_now", "metric": "asleep_auc", "max_gap": 0.1}], ["sex"], opts())


# ---- reweighing ---------------------------------------------------------------------------------

def test_reweighing_makes_group_and_answer_independent():
    rng = np.random.default_rng(1)
    group = np.where(rng.random(400) < 0.3, "male", "female")
    # OFF is rarer among men: 20% against 50%.
    off = np.where(group == "male", rng.random(400) < 0.2, rng.random(400) < 0.5)
    labels = pd.DataFrame({"state_now": np.where(off, 2.0, 3.0)})
    slices = pd.DataFrame({"sex": group})
    w = reweighing_weights(labels, slices, [{"label": "state_now", "metric": "off_recall", "column": "sex"}],
                           max_weight=100)
    weighted_off = {g: w[(group == g) & off].sum() / w[group == g].sum() for g in ("male", "female")}
    assert weighted_off["male"] == pytest.approx(weighted_off["female"]) == pytest.approx(off.mean())


def test_unanswered_hours_keep_weight_1_and_weights_are_capped():
    labels = pd.DataFrame({"state_now": [2.0, 3.0, 3.0, 3.0, np.nan]})
    slices = pd.DataFrame({"sex": ["male", "female", "female", "female", "male"]})
    failure = [{"label": "state_now", "metric": "off_recall", "column": "sex"}]
    w = reweighing_weights(labels, slices, failure, max_weight=2.0)
    assert w[4] == 1.0 and w.max() <= 2.0 and w.min() >= 0.5
    assert np.all(reweighing_weights(labels, slices, failure, power=0) == 1.0)


# ---- the step, from train's outputs ---------------------------------------------------------------

def test_a_gap_from_imbalance_fails_the_gate_alerts_and_reweighing_fixes_it(tmp_path, alerts):
    # OFF is rare in DBS patients and the model reads DBS, so it misses their OFF hours.
    cfg, data, candidates = trained(tmp_path, kind="imbalance", seed=1)
    tracker = MemoryTracker()
    out = CheckBias(cfg, tracker).run(candidates, data=data, split=SPLIT)

    original, mitigated = out
    assert original["name"] == "fake_hourly" and original["gates"]["bias"] is False
    assert any("by dbs" in f for f in original["bias"]["failed"])
    assert mitigated["name"] == "fake_hourly_reweighted" and mitigated["mitigated_from"] == "fake_hourly"
    assert mitigated["gates"]["bias"] is True
    assert registry.load(mitigated["model_dir"]).manifest()["training"]["weighted"] is True
    assert "cv/state_now/off_auc" in mitigated["metrics"]
    assert alerts and "Bias gate failed" in alerts[0][0] and "dbs" in alerts[0][1]

    folder = tmp_path / "bias"
    report = json.loads((folder / "bias.json").read_text(encoding="utf-8"))
    by_dbs = report["candidates"]["fake_hourly"]["slices"]["state_now"]["dbs"]
    assert by_dbs["no"]["metrics"]["off_recall"] > by_dbs["yes"]["metrics"]["off_recall"]
    assert by_dbs["yes"]["participants"] == 4 and by_dbs["no"]["participants"] == 6
    [gap] = report["mitigation"]["fake_hourly"]["gaps"]
    assert gap["check"] == "state_now/off_recall by dbs" and gap["after"] < 0.1 < gap["before"] and gap["fixed"]
    assert "cv/state_now/off_auc" in report["mitigation"]["fake_hourly"]["selection_metric"]
    text = (folder / "bias_report.md").read_text(encoding="utf-8")
    assert "# Bias report" in text and "Trade-off" in text and "**fail**" in text
    assert (folder / "plots" / "fake_hourly__state_now__off_recall.png").stat().st_size > 0
    saved = json.loads((tmp_path / "candidates.json").read_text(encoding="utf-8"))   # the shared file
    assert [c["name"] for c in saved] == ["fake_hourly", "fake_hourly_reweighted"]
    assert [c["gates_passed"] for c in saved] == [False, True]                       # what select reads
    assert not (folder / "candidates.json").exists()

    run = tracker.find("bias")
    assert run.tags["owner"] == "P3"
    names = [r.name for r in tracker.children(run.run_id)]
    assert names[:2] == ["bias-fake_hourly", "fake_hourly_reweighted"]
    logged = tracker.find("bias-fake_hourly").latest("cv/state_now/bias_gap_off_recall_dbs")
    assert logged == pytest.approx(gap["before"])


def test_reweighing_cant_fix_a_gap_from_weaker_signal_so_the_release_is_blocked(tmp_path, alerts):
    # OFF is as common in men as in women, but their movement shows it less: nothing to rebalance.
    cfg, data, candidates = trained(tmp_path, kind="signal", seed=1)
    with pytest.raises(GateError, match="even after mitigation"):
        CheckBias(cfg, MemoryTracker()).run(candidates, data=data, split=SPLIT)
    report = json.loads((tmp_path / "bias" / "bias.json").read_text(encoding="utf-8"))
    [gap] = report["mitigation"]["fake_hourly"]["gaps"]
    assert gap["check"] == "state_now/off_recall by sex" and not gap["fixed"]
    assert abs(gap["after"] - gap["before"]) < 0.05
    assert "still fails" in (tmp_path / "bias" / "bias_report.md").read_text(encoding="utf-8")


def test_mitigation_weights_come_from_each_folds_training_patients_only(tmp_path, alerts, monkeypatch):
    cfg, data, candidates = trained(tmp_path, kind="imbalance", seed=1)
    seen = []
    real = bias.reweighing_weights
    monkeypatch.setattr(bias, "reweighing_weights", lambda labels, slices, *a, **k: (
        seen.append(set(slices["participant_id"])), real(labels, slices, *a, **k))[1])
    CheckBias(cfg, MemoryTracker()).run(candidates, data=data, split=SPLIT)
    folds = [set(f) for f in SPLIT.folds]
    for k, people in enumerate(seen[:len(folds)]):
        assert people == set(SPLIT.development) - folds[k]          # never the fold's validation patients
    assert seen[-1] == set(SPLIT.development)                       # the final refit: every development patient
    assert not any(set(SPLIT.test) & s for s in seen)


def test_a_fair_enough_model_passes_without_mitigation(tmp_path, alerts):
    cfg, data, candidates = trained(tmp_path, max_recall_gap=1.0)
    out = CheckBias(cfg, MemoryTracker()).run(candidates, data=data, split=SPLIT)
    assert [c["name"] for c in out] == ["fake_hourly"] and out[0]["gates"]["bias"] is True
    assert alerts == []
    assert "No candidate needed mitigation" in (tmp_path / "bias" / "bias_report.md").read_text(encoding="utf-8")


def test_no_candidate_passing_blocks_the_release(tmp_path, alerts):
    cfg, data, candidates = trained(tmp_path, mitigation={"enabled": False})
    with pytest.raises(GateError, match="bias gate failed"):
        CheckBias(cfg, MemoryTracker()).run(candidates, data=data, split=SPLIT)
    assert (tmp_path / "bias" / "bias.json").exists() and alerts       # reported and alerted before failing


def test_other_gates_on_a_candidate_are_kept(tmp_path, alerts):
    cfg, data, candidates = trained(tmp_path, max_recall_gap=1.0)
    candidates[0]["gates"] = {"validation": True}
    out = CheckBias(cfg, MemoryTracker()).run(candidates, data=data, split=SPLIT)
    assert out[0]["gates"] == {"validation": True, "bias": True} and out[0]["gates_passed"] is True


def test_gates_passed_needs_every_gate_so_a_failed_validation_still_blocks(tmp_path, alerts):
    cfg, data, candidates = trained(tmp_path, max_recall_gap=1.0)
    candidates[0]["gates"] = {"validation": False}
    out = CheckBias(cfg, MemoryTracker()).run(candidates, data=data, split=SPLIT)   # bias itself passes
    assert out[0]["gates"] == {"validation": False, "bias": True} and out[0]["gates_passed"] is False


def test_a_validation_failure_set_by_evaluate_is_never_overwritten(tmp_path, alerts):
    cfg, data, candidates = trained(tmp_path, max_recall_gap=1.0)
    candidates[0]["gates_passed"] = False                           # how evaluate (P1) reports a failure
    out = CheckBias(cfg, MemoryTracker()).run(candidates, data=data, split=SPLIT)
    assert out[0]["gates"] == {"validation": False, "bias": True} and out[0]["gates_passed"] is False


def test_the_reweighted_candidate_is_held_to_the_validation_gate(tmp_path, alerts):
    cfg, data, candidates = trained(tmp_path, kind="imbalance", seed=1)
    cfg["validation_gate"] = {"min": {"cv/state_now/off_auc": 0.99}}  # higher than any model here
    CheckBias(cfg, MemoryTracker()).run(candidates, data=data, split=SPLIT)    # bias passes; select blocks it
    saved = json.loads((tmp_path / "candidates.json").read_text(encoding="utf-8"))
    mitigated = saved[1]
    assert mitigated["gates"] == {"validation": False, "bias": True} and mitigated["gates_passed"] is False
    assert "cv/state_now/off_auc" in mitigated["validation_failures"][0]


def test_a_gate_that_judged_nothing_blocks_and_isnt_mitigated(tmp_path, alerts):
    # Every group needs 7 patients; none has that many, so no check can be judged.
    cfg, data, candidates = trained(tmp_path, max_recall_gap=1.0, min_slice_participants=7)
    with pytest.raises(GateError, match="no candidate passed"):
        CheckBias(cfg, MemoryTracker()).run(candidates, data=data, split=SPLIT)
    saved = json.loads((tmp_path / "candidates.json").read_text(encoding="utf-8"))
    assert [c["name"] for c in saved] == ["fake_hourly"]            # no pointless reweighted retrain
    assert saved[0]["gates"]["bias"] is False and saved[0]["bias"]["judged"] == 0
    assert any("nothing could be judged" in body for _, body in alerts)
    assert "fail** (nothing judged)" in (tmp_path / "bias" / "bias_report.md").read_text(encoding="utf-8")


def test_missing_predictions_say_to_run_train(tmp_path):
    cfg = pipeline_config(tmp_path)
    with pytest.raises(SchemaError, match="run train first"):
        CheckBias(cfg, MemoryTracker()).run([{"name": "x"}], data=biased_dataset(hours=3), split=SPLIT)
