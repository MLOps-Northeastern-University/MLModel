import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from model_pipeline.core.errors import ConfigError, GateError, SchemaError
from model_pipeline.data.folds import Split
from model_pipeline.models import registry
from model_pipeline.steps.train import TrainModels
from model_pipeline.tracking.memory_tracker import MemoryTracker
from tests.models.test_window_classifier import dataset, environment  # noqa: F401  (the fake harnet10)

PIDS = ("A", "B", "C", "D", "E", "F", "G")
SPLIT = Split(["G"], [["A", "B"], ["C", "D"], ["E", "F"]], seed=0)
WINDOWS = {"spec_version": 1, "seconds": 10, "rate_hz": 30, "axes": 3}
TINY_TRANSFORMER = {"network": {"movement_params": {"hidden_dim": 16, "output_dim": 16}, "d_model": 16,
                                "dropout": 0.1, "decoder": {"heads": 2, "query_block": 32}, "heads": {"hidden": 8}},
                    "pretrain": {"epochs": 1, "chunk_h": 4},
                    "diary": {"max_epochs": 3, "patience": 1, "chunk_h": 4, "eval_past_h": [1]}, "score_past_h": 2}


def data():
    d = dataset(pids=PIDS)
    patients = pd.DataFrame({"participant_id": list(PIDS), "more_affected_side": True,
                             "dbs": [i % 2 == 0 for i in range(len(PIDS))]})
    events = pd.DataFrame([{"participant_id": "A", "time_s": 2 * 3600 + 30.0, "kind": "dose", "ledd": 100.0,
                            "source": "on_time", "kinesia_score": np.nan, "tremor_score": np.nan}])
    return replace(d, patients=patients, events=events)


def config(tmp_path, *entries, score_test=False, leakage=("window_logistic",), max_auc=0.99):
    entries = entries or ({"name": "window_logistic", "family": "window_logistic"},)
    return {"windows": WINDOWS, "resolved_paths": {"artifacts_dir": tmp_path},
            "models": {"targets": ["sleep", "state", "tremor"], "horizons": ["now", "next_hour"],
                       "candidates": list(entries)},
            "train": {"seeds": [0], "score_test": score_test,
                      "leakage_check": {"candidates": list(leakage), "max_auc": max_auc}}}


def test_every_candidate_is_cross_validated_refit_and_tracked(tmp_path):
    tracker = MemoryTracker()
    boosting = {"name": "window_boosting", "family": "window_boosting",
                "params": {"pca_components": 8, "iterations": 20}}
    cfg = config(tmp_path, {"name": "window_logistic", "family": "window_logistic"}, boosting)
    results = TrainModels(cfg, tracker).run(data(), SPLIT)
    assert [r["name"] for r in results] == ["window_logistic", "window_boosting"]

    train = tracker.find("train")
    assert train.tags["split_hash"] == SPLIT.hash and train.tags["test_scored"] == "false"
    names = [r.name for r in tracker.children(train.run_id)]
    assert names == ["leakage_check-window_logistic", "window_logistic", "window_boosting"]
    for r in results:
        run = tracker.runs[r["run_id"]]
        assert [c.name for c in tracker.children(run.run_id)] == ["seed-0-fold-1", "seed-0-fold-2", "seed-0-fold-3",
                                                                  "final"]
        fold = tracker.children(run.run_id)[0]
        assert 0 <= fold.latest("val/state_now/off_auc_within") <= 1 and fold.latest("run/fit_s") > 0
        assert r["metrics"]["cv/state_now/off_auc"] > 0.9                # the synthetic movement shows the state
        assert "cv/state_now/off_auc_fold_sd" in run.metrics and "cv/state_now/off_auc_no_dbs" in r["metrics"]
        assert not any(k.startswith("test/") for k in r["metrics"])
        model = registry.load(r["model_dir"])                             # the final model, saved and loadable
        assert model.is_fitted and model.manifest()["training"]["train_participants"] == 6
        final = tracker.children(run.run_id)[-1]
        assert final.artifacts and final.artifacts[0][1] == f"model/{r['name']}"

    saved = json.loads((tmp_path / "candidates.json").read_text(encoding="utf-8"))
    assert [c["name"] for c in saved] == ["window_logistic", "window_boosting"]


def test_the_predictions_files_hold_every_validation_hour_and_never_the_test_patient(tmp_path):
    TrainModels(config(tmp_path), MemoryTracker()).run(data(), SPLIT)
    state = pd.read_csv(tmp_path / "predictions" / "state_now.csv")
    assert set(state.participant_id) == set("ABCDEF") and len(state) == 6 * 10
    assert set(state.fold.astype(str)) == {"1", "2", "3"} and set(state.model) == {"window_logistic"}
    assert (~state.worn).sum() == 6                                       # hour 4 of each patient is unworn
    assert list(state.columns[-7:]) == ["severe_akinesia", "discomforting_akinesia", "slight_akinesia",
                                        "good_kinesia", "slight_dyskinesia", "discomforting_dyskinesia",
                                        "severe_dyskinesia"]
    np.testing.assert_allclose(state.iloc[:, -7:].sum(axis=1), 1)
    assert {p.name for p in (tmp_path / "predictions").iterdir()} == {
        f"{t}_{h}.csv" for t in ("sleep", "state", "tremor") for h in ("now", "next_hour")}


def test_the_test_patient_is_scored_only_when_asked(tmp_path):
    tracker = MemoryTracker()
    [result] = TrainModels(config(tmp_path, score_test=True), tracker).run(data(), SPLIT)
    assert tracker.find("train").tags["test_scored"] == "true"
    assert "test/state_now/off_auc" in result["metrics"]
    state = pd.read_csv(tmp_path / "predictions" / "state_now.csv")
    assert set(state.loc[state.fold.astype(str) == "test", "participant_id"]) == {"G"}


def test_the_final_transformer_trains_the_median_of_the_folds_best_epochs(tmp_path):
    tracker = MemoryTracker()
    cfg = config(tmp_path, {"name": "mct", "family": "motor_causal_transformer", "params": TINY_TRANSFORMER},
                 leakage=())
    [result] = TrainModels(cfg, tracker).run(data(), SPLIT)
    run = tracker.runs[result["run_id"]]
    folds = tracker.children(run.run_id)[:3]
    best = [registry.load(result["model_dir"]).training_result["best_epoch"]]
    expected = int(np.floor(np.median([_best(f) for f in folds]) + 0.5))
    assert result["final_params"] == {"final_epochs": expected} == {"final_epochs": best[0]}
    assert tracker.children(run.run_id)[-1].tags["final_epochs"] == str(expected)
    curve = folds[0].metrics["run/diary_train_loss"]
    assert [step for step, _ in curve] == list(range(1, len(curve) + 1))   # one value per epoch
    # Pretraining: each fold's validation skill per epoch, and the folds' last values averaged on the candidate.
    skill = "run/pretrain_val_ahead_120s_skill"
    assert all(len(f.metrics[skill]) == 1 for f in folds)                  # 1 pretraining epoch
    assert run.latest(skill) == pytest.approx(np.mean([f.latest(skill) for f in folds]))
    assert "run/pretrain_val_skill" in run.metrics and run.latest("run/pretrain_seconds") > 0
    assert run.tags["split_hash"] == SPLIT.hash


def test_a_candidate_without_pretraining_logs_no_pretraining(tmp_path):
    tracker = MemoryTracker()
    params = {**TINY_TRANSFORMER, "pretrain": {"epochs": 0}}
    cfg = config(tmp_path, {"name": "mct_no_pretrain", "family": "motor_causal_transformer", "params": params},
                 leakage=())
    [result] = TrainModels(cfg, tracker).run(data(), SPLIT)
    run = tracker.runs[result["run_id"]]
    assert run.params["params.pretrain.epochs"] == 0
    assert not any(m.startswith("run/pretrain") for r in [run, *tracker.children(run.run_id)] for m in r.metrics)


def _best(fold_run):
    """The best epoch of a fold, from its curve: where the within-patient AUC first peaked."""
    curve = fold_run.metrics["run/diary_within_auc"]
    values = [v for _, v in curve]
    return curve[int(np.nanargmax(values))][0]


def test_a_model_that_scores_on_shuffled_labels_stops_the_run(tmp_path):
    cfg = config(tmp_path, max_auc=0.501)          # so tight that chance-level noise counts as a leak
    tracker = MemoryTracker()
    with pytest.raises(GateError, match="leakage gate failed: window_logistic trained on shuffled labels"):
        TrainModels(cfg, tracker).run(data(), SPLIT)
    assert tracker.find("train").status.value == "FAILED"
    assert not (tmp_path / "candidates.json").exists()


def test_the_data_must_name_the_splits_patients(tmp_path):
    with pytest.raises(SchemaError, match="aren't in the split"):
        TrainModels(config(tmp_path), MemoryTracker()).run(data(), Split(["G"], [["A", "B"], ["C", "D"]], seed=0))


@pytest.mark.parametrize("train, match", [
    ({"seeds": [], "score_test": False}, "seeds"),
    ({"seeds": [0, 0], "score_test": False}, "seeds"),
    ({"seeds": [0], "score_test": "no"}, "score_test"),
    ({"seeds": [0], "score_test": False, "leakage_check": {"candidates": ["nope"], "max_auc": 0.6}}, "unknown"),
    ({"seeds": [0], "score_test": False, "leakage_check": {"candidates": ["window_logistic"], "max_auc": 0.4}},
     "max_auc"),
])
def test_bad_train_settings_are_refused(tmp_path, train, match):
    cfg = {**config(tmp_path), "train": train}
    with pytest.raises(ConfigError, match=match):
        TrainModels(cfg, MemoryTracker())
    with pytest.raises(ConfigError, match="train section"):
        TrainModels({k: v for k, v in cfg.items() if k != "train"}, MemoryTracker())


def test_it_loads_the_split_and_only_the_patients_it_needs_from_files(tmp_path):
    from model_pipeline.data.folds import write_split
    from tests.data.test_loaders import write_files
    from tests.helpers import make_config
    paths = write_files(tmp_path, people=["COPS-1", "COPS-2", "COPS-3"])
    paths["split_file"] = str(tmp_path / "split.json")
    write_split(Split(["COPS-3"], [["COPS-1"], ["COPS-2"]], seed=0), paths["split_file"])
    cfg = make_config(tmp_path, features_csv=paths["features_csv"], paths=paths,
                      models={"targets": ["tremor"], "horizons": ["now"],
                              "candidates": [{"name": "window_logistic", "family": "window_logistic"}]},
                      train={"seeds": [0], "score_test": False, "use_tuned": False,
                             "leakage_check": {"candidates": []}})
    tracker = MemoryTracker()
    [result] = TrainModels(cfg, tracker).run()
    assert len(tracker.find("train").tags["data_fingerprint"]) == 16
    state = pd.read_csv(cfg["resolved_paths"]["artifacts_dir"] / "predictions" / "tremor_now.csv")
    assert set(state.participant_id) == {"COPS-1", "COPS-2"}           # the test patient wasn't even loaded
    assert result["final_params"] == {}


def test_the_data_fingerprint_covers_every_table_and_the_window_files_contents(tmp_path):
    from model_pipeline.experiments import data_fingerprint
    from tests.data.test_loaders import write_files
    paths = write_files(tmp_path)
    first = data_fingerprint(paths)
    assert data_fingerprint(paths) == first                                    # whichever step asks
    signal = tmp_path / "windows" / "COPS-1" / "signal.npy"
    values = np.load(signal)
    np.save(signal, values + 1.0)                                              # same size, new contents
    assert data_fingerprint(paths) != first
    np.save(signal, values)
    assert data_fingerprint(paths) == first
    events = pd.read_csv(paths["events_csv"]).assign(ledd=50.0)
    events.to_csv(paths["events_csv"], index=False)
    assert data_fingerprint(paths) != first                                    # a table a baseline never reads


def test_the_test_patients_are_scored_once_per_split(tmp_path):
    TrainModels(config(tmp_path, score_test=True, leakage=()), MemoryTracker()).run(data(), SPLIT)
    assert (tmp_path / "test_scored" / f"{SPLIT.hash}.json").exists()
    with pytest.raises(ConfigError, match="already scored"):
        TrainModels(config(tmp_path, score_test=True, leakage=()), MemoryTracker()).run(data(), SPLIT)
    TrainModels(config(tmp_path, leakage=()), MemoryTracker()).run(data(), SPLIT)    # without the test: fine


def test_a_failed_run_leaves_no_outputs_from_the_last_one(tmp_path):
    TrainModels(config(tmp_path, leakage=()), MemoryTracker()).run(data(), SPLIT)
    assert (tmp_path / "candidates.json").exists() and list((tmp_path / "predictions").glob("*.csv"))
    with pytest.raises(GateError):
        TrainModels(config(tmp_path, max_auc=0.501), MemoryTracker()).run(data(), SPLIT)
    assert not (tmp_path / "candidates.json").exists() and not list((tmp_path / "predictions").glob("*.csv"))


def test_runs_record_the_code_they_came_from(tmp_path):
    tracker = MemoryTracker()
    TrainModels(config(tmp_path, leakage=()), tracker).run(data(), SPLIT)
    train = tracker.find("train")
    assert {"git_commit", "git_dirty"} <= set(train.tags)
    assert "environment.packages.numpy" in train.params and "environment.python" in train.params


def test_every_run_with_predictions_gets_its_charts_and_the_train_run_the_comparison(tmp_path):
    tracker = MemoryTracker()
    TrainModels(config(tmp_path, score_test=True), tracker).run(data(), SPLIT)
    charts = tmp_path / "charts"
    train = tracker.find("train")
    assert (charts / "comparison", "charts") in train.artifacts
    assert (charts / "comparison" / "comparison.png").exists()
    candidate = tracker.find("window_logistic")
    assert (charts / "window_logistic" / "cv", "charts") in candidate.artifacts
    runs = {r.name: r for r in tracker.children(candidate.run_id)}
    assert (charts / "window_logistic" / "seed-0-fold-2", "charts") in runs["seed-0-fold-2"].artifacts
    assert runs["final"].artifacts[0][1] == "model/window_logistic"      # the model first, then the test charts
    assert (charts / "window_logistic" / "test", "charts") in runs["final"].artifacts
    names = sorted(p.name for p in (charts / "window_logistic" / "cv").iterdir())
    assert names == sorted(["scorecard.png"] + [f"confusion_{t}_{h}.png" for t in ("sleep", "state", "tremor")
                                                for h in ("now", "next_hour")])


def test_charts_can_be_turned_off_or_kept_off_the_folds(tmp_path):
    cfg = config(tmp_path)
    cfg["train"]["charts"] = {"folds": False}
    tracker = MemoryTracker()
    TrainModels(cfg, tracker).run(data(), SPLIT)
    fold = tracker.find("seed-0-fold-1")
    assert fold.artifacts == [] and (tmp_path / "charts" / "window_logistic" / "cv").exists()
    cfg["train"]["charts"] = {"enabled": False}
    TrainModels(cfg, MemoryTracker()).run(data(), SPLIT)
    assert not (tmp_path / "charts").exists()                           # the last run's charts are cleared too


def test_the_leakage_check_judges_only_its_targets(tmp_path, monkeypatch):
    from model_pipeline.steps import train as train_module

    def scores(_):        # the shuffled-label model: sleep far from chance, state and tremor at chance
        return {"sleep_now": {"asleep_auc": 0.75, "asleep_auc_within": 0.78},
                "state_now": {"off_auc": 0.51, "off_auc_within": 0.49},
                "tremor_now": {"any_tremor_auc": 0.52}}

    monkeypatch.setattr(train_module, "score", scores)
    cfg = config(tmp_path, max_auc=0.6)
    with pytest.raises(GateError, match="sleep_now/asleep_auc 0.750"):        # every target: sleep stops the run
        TrainModels(cfg, MemoryTracker()).run(data(), SPLIT)
    cfg["train"]["leakage_check"]["targets"] = ["state", "tremor"]
    TrainModels(cfg, MemoryTracker()).run(data(), SPLIT)                     # sleep isn't judged: it passes
    monkeypatch.setattr(train_module, "score", lambda _: {"state_now": {"off_auc_within": 0.7}})
    with pytest.raises(GateError, match="state_now/off_auc_within 0.700"):
        TrainModels(cfg, MemoryTracker()).run(data(), SPLIT)


def test_the_leakage_check_refuses_unknown_targets(tmp_path):
    cfg = config(tmp_path)
    cfg["train"]["leakage_check"]["targets"] = ["mood"]
    with pytest.raises(ConfigError, match="leakage_check.targets"):
        TrainModels(cfg, MemoryTracker())


def test_within_only_judges_only_the_within_patient_aucs(tmp_path, monkeypatch):
    from model_pipeline.steps import train as train_module

    def scores(_):        # pooled AUC far from chance (patients differ), within-patient at chance
        return {"state_now": {"off_auc": 0.36, "off_auc_within": 0.51}, "tremor_now": {"any_tremor_auc_within": 0.48}}

    monkeypatch.setattr(train_module, "score", scores)
    cfg = config(tmp_path, max_auc=0.6)
    cfg["train"]["leakage_check"]["targets"] = ["state", "tremor"]
    with pytest.raises(GateError, match="state_now/off_auc 0.360"):           # pooled judged: it stops
        TrainModels(cfg, MemoryTracker()).run(data(), SPLIT)
    cfg["train"]["leakage_check"]["within_only"] = True
    TrainModels(cfg, MemoryTracker()).run(data(), SPLIT)                     # within only: it passes
    monkeypatch.setattr(train_module, "score", lambda _: {"state_now": {"off_auc_within": 0.7}})
    with pytest.raises(GateError, match="state_now/off_auc_within 0.700"):  # a within-patient leak still stops it
        TrainModels(cfg, MemoryTracker()).run(data(), SPLIT)
