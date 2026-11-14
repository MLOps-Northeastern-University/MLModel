"""The explain step: exact Shapley values over input groups, leaving groups out, and a full run."""

import json
import math

import numpy as np
import pytest

from scripts.core.data import Windows
from scripts.core.errors import ConfigError
from scripts.steps.explain import ExplainModels, coalitions, leave_out, settings, shapley
from scripts.tracking.memory_tracker import MemoryTracker
from tests.steps.fake_models import SPLIT, HourlyLogistic, biased_dataset, pipeline_config


def test_shapley_values_add_up_to_the_full_effect():
    rng = np.random.default_rng(0)
    groups = ["windows", "events", "patients"]
    values = {c: rng.random(5) for c in coalitions(groups)}
    phi = shapley(values, groups)
    total = sum(phi.values())
    assert np.allclose(total, values[frozenset(groups)] - values[frozenset()])


def test_shapley_values_of_an_additive_model_are_each_groups_own_effect():
    effect = {"windows": 0.3, "events": -0.1, "patients": 0.05}
    values = {c: np.array([sum(effect[g] for g in c)]) for c in coalitions(list(effect))}
    phi = shapley(values, list(effect))
    assert {g: float(v[0]) for g, v in phi.items()} == pytest.approx(effect)


def test_a_group_the_output_ignores_gets_nothing():
    values = {c: np.array([0.4 if "windows" in c else 0.1]) for c in coalitions(["windows", "events"])}
    phi = shapley(values, ["windows", "events"])
    assert phi["events"][0] == 0.0 and phi["windows"][0] == pytest.approx(0.3)


def test_leaving_out_events_removes_them_and_patients_swaps_fields():
    data = biased_dataset(hours=3)
    rng = np.random.default_rng(0)
    assert len(leave_out(data, "events", rng).events) == 0
    one = data.subset(["P01"])
    swapped = leave_out(one, "patients", rng, donors=data.patients).patients
    assert swapped["participant_id"].tolist() == ["P01"]
    donor = data.patients[data.patients["participant_id"] != "P01"]
    row = swapped.iloc[0]
    assert ((donor["dbs"] == row["dbs"]) & (donor["more_affected_side"] == row["more_affected_side"])).any()


def test_leaving_out_features_shuffles_each_patients_own_hours():
    data = biased_dataset(hours=6)
    out = leave_out(data, "features", np.random.default_rng(0)).features
    for pid in data.participants:
        rows = (data.hours["participant_id"] == pid).to_numpy()
        assert sorted(out[rows]["move_mean"]) == sorted(data.features[rows]["move_mean"])
    assert not out["move_mean"].equals(data.features["move_mean"])


def test_leaving_out_movement_shuffles_worn_windows_only():
    from dataclasses import replace
    data = biased_dataset(hours=1).subset(["P01"])
    signal = np.arange(6 * 3 * 300, dtype=np.float32).reshape(6, 3, 300)
    worn = np.array([True, True, True, True, False, False])
    data = replace(data, windows={"P01": Windows(time_s=np.arange(6) * 10, worn=worn, signal=signal)})
    out = leave_out(data, "windows", np.random.default_rng(3)).windows["P01"]
    assert np.array_equal(out.signal[~worn], signal[~worn]) and np.array_equal(out.time_s, np.arange(6) * 10)
    assert sorted(out.signal[worn, 0, 0]) == sorted(signal[worn, 0, 0])
    assert data.windows["P01"].signal is signal                                   # the original is untouched


def test_off_has_to_be_quoted_in_the_config(tmp_path):
    cfg = pipeline_config(tmp_path)
    cfg["explain"]["outputs"] = [{"label": "state_now", "question": False}]       # what YAML makes of off
    with pytest.raises(ConfigError, match="quote"):
        settings(cfg)


def test_a_run_explains_each_output_by_input_group(tmp_path):
    cfg = pipeline_config(tmp_path)
    data = biased_dataset()
    model = HourlyLogistic(["sleep", "state", "tremor"], ["now"]).fit(data.subset(SPLIT.development), seed=0)
    tracker = MemoryTracker()
    results = ExplainModels(cfg, tracker).run([{"name": "fake_hourly", "model_dir": "unused"}], data=data,
                                              split=SPLIT, models={"fake_hourly": model})

    r = results["fake_hourly"]
    assert r["groups"] == ["events", "patients", "features"] and len(r["participants"]) == 4
    assert not set(r["participants"]) & set(SPLIT.test)
    off = r["outputs"]["state_now/off"]
    shap = {g: v["mean_abs_shap"] for g, v in off["groups"].items()}
    assert shap["features"] > shap["events"] and shap["features"] > shap["patients"]  # movement shows OFF
    assert off["ablation"]["without"]["features"]["auc_drop"] > 0.1
    assert set(off["by_slice"]["sex"]) == {"female", "male"}
    total = sum(v["mean_shap"] for v in off["groups"].values())
    assert math.isclose(total, off["mean_prediction"] - off["base"], abs_tol=1e-9)

    folder = tmp_path / "explain" / "fake_hourly"
    assert json.loads((folder / "shap.json").read_text(encoding="utf-8"))["candidate"] == "fake_hourly"
    assert (folder / "shap.png").stat().st_size > 0 and (folder / "ablation.png").stat().st_size > 0
    assert "# What drives the models" in (tmp_path / "explain" / "explain_report.md").read_text(encoding="utf-8")
    run = tracker.find("explain-fake_hourly")
    assert run.latest("train/state_now/shap_off_features") == pytest.approx(shap["features"])


def test_only_the_listed_candidates_are_explained(tmp_path):
    cfg = pipeline_config(tmp_path)
    cfg["explain"]["candidates"] = ["nothing_like_it"]
    with pytest.raises(ConfigError, match="none of explain.candidates"):
        ExplainModels(cfg, MemoryTracker()).run([{"name": "fake_hourly", "model_dir": "x"}],
                                                data=biased_dataset(hours=3), split=SPLIT, models={})
