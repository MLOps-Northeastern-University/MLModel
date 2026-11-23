"""The checks: a broken contract stops the run; warnings and the bias report don't."""

import shutil

import numpy as np
import pandas as pd
import pytest

from data_pipeline import checks, pipeline
from data_pipeline.utils import read_table, write_json, write_table
from tests.helpers import make_local_run


@pytest.fixture
def cfg(tmp_path):
    cfg = make_local_run(tmp_path)
    assert pipeline.run(cfg) == 0
    return cfg


def edit(cfg, name, change):
    path = cfg["resolved_paths"]["processed_dir"] / name
    write_table(change(read_table(path)), path)


def fails(cfg, match):
    with pytest.raises(checks.DataCheckError, match=match):
        checks.run_checks(cfg)


def test_a_good_dataset_passes(cfg):
    assert checks.run_checks(cfg)["critical"] == []
    md = (cfg["resolved_paths"]["reports_dir"] / "bias_report.md").read_text()
    assert "## sex" in md and "asleep" in md


def test_a_repeated_hour_stops_the_run(cfg):
    edit(cfg, "hours.csv", lambda h: pd.concat([h, h.iloc[[0]]]))
    fails(cfg, "participant-hour twice")


def test_bad_scores_stop_the_run(cfg):
    edit(cfg, "hours.csv", lambda h: h.assign(kinesia_score=h["kinesia_score"].fillna(0) + 10))
    fails(cfg, "kinesia_score outside")


def test_a_bad_dose_stops_the_run(cfg):
    edit(cfg, "events.csv", lambda e: e.assign(ledd=-5.0))
    fails(cfg, "LEDD of 0 or more")


def test_a_patient_without_dbs_stops_the_run(cfg):
    edit(cfg, "patients.csv", lambda p: p.assign(dbs=np.nan))
    fails(cfg, "dbs isn't true or false")


def test_missing_columns_stop_the_run(cfg):
    edit(cfg, "events.csv", lambda e: e.drop(columns="source"))
    fails(cfg, "missing columns")


def test_windows_cut_with_another_spec_stop_the_run(cfg):
    cfg["windows"]["rate_hz"] = 25
    fails(cfg, "cut with spec")


def test_broken_windows_stop_the_run(cfg):
    folder = cfg["resolved_paths"]["processed_dir"] / "windows" / "COPS-1"
    np.save(folder / "signal.npy", np.load(folder / "signal.npy").astype(np.float64))
    shutil.rmtree(cfg["resolved_paths"]["processed_dir"] / "windows" / "COPS-2")
    fails(cfg, "float32")
    fails(cfg, "COPS-2: windows are missing")


def test_windows_outside_the_diary_stop_the_run(cfg):
    folder = cfg["resolved_paths"]["processed_dir"] / "windows" / "COPS-1"
    np.save(folder / "time_s.npy", np.load(folder / "time_s.npy") + 100 * 3600)
    fails(cfg, "outside the diary hours")


def test_thin_participants_and_disagreeing_wrists_are_warnings(cfg):
    cfg["checks"]["min_labelled_hours"] = 100
    reports = cfg["resolved_paths"]["reports_dir"]
    write_table(pd.DataFrame([{"participant_id": "COPS-1", "minutes": 50, "corr_x": -0.4, "corr_y": 0.5, "corr_z": 0.2},
                              {"participant_id": "COPS-2", "minutes": 0}]), reports / "wrist_agreement.csv")
    write_json([{"field": "kinesia", "issue": "score_without_label"}], reports / "label_issues.json")
    warnings = checks.run_checks(cfg)["warnings"]
    assert any("usable hours (minimum 100)" in w for w in warnings)
    assert any("COPS-1: wrists still disagree" in w for w in warnings)
    assert any("COPS-2: wrist agreement not checked" in w for w in warnings)
    assert "diary kinesia: 1 x score_without_label" in warnings


def test_states_come_from_the_score_and_sleep():
    hours = pd.DataFrame({"kinesia_score": [-2, 0, 3, np.nan, np.nan],
                          "kinesia_text": ["", "", "", "Sleep", "No Data"]})
    states = checks.state(hours)
    assert states[:4].tolist() == ["off", "on", "dyskinetic", "asleep"] and pd.isna(states[4])
