"""The whole pipeline, as `dvc repro` and `python -m scripts.run_pipeline` run it.

Synthetic participants on disk stand in for OSF (data.local_zip_dir), so
nothing is downloaded. Each one has 7 diary hours: 4 usable (on, off and two
dyskinetic), plus a sleep hour, a "No Data" hour and an hour the watch wasn't worn.
"""

import zipfile

import pandas as pd
import pytest

from scripts import pipeline_steps, run_pipeline
from scripts.anomalies import DataAnomalyError
from scripts.utils.file_io import read_json
from tests.helpers import make_config, make_demographics, make_participant_zip

PIDS = ["COPS-1", "COPS-2", "COPS-3", "COPS-4"]


@pytest.fixture
def alerts(monkeypatch):
    """Record alerts instead of sending them."""
    sent = []

    def record(subject, body, cfg=None):
        sent.append(subject)

    monkeypatch.setattr(run_pipeline, "notify", record)
    monkeypatch.setattr(pipeline_steps, "notify", record)
    return sent


def participants(tmp_path, pids=PIDS):
    folder = tmp_path / "zips"
    folder.mkdir()
    for pid in pids:
        make_participant_zip(folder, pid)
    make_demographics(folder, pids)
    return folder


def use_config(monkeypatch, cfg):
    """Make run_pipeline.main use this config instead of config/pipeline.yaml."""
    monkeypatch.setattr(run_pipeline, "load_config", lambda path=None, overrides=None: cfg)
    return cfg


def outputs(cfg):
    return cfg["resolved_paths"]["processed_dir"], cfg["resolved_paths"]["reports_dir"]


def test_the_whole_pipeline_runs_and_writes_every_output(tmp_path, monkeypatch, alerts):
    zips = participants(tmp_path)
    cfg = use_config(monkeypatch, make_config(
        tmp_path / "project", data={"local_zip_dir": str(zips)}, run={"parallel_participants": 2}))

    assert run_pipeline.main([]) == 0

    processed, reports = outputs(cfg)
    features = pd.read_csv(processed / "hourly_features.csv")
    assert len(features) == 16  # 4 usable hours x 4 participants
    assert sorted(features["participant_id"].unique()) == PIDS
    assert set(features["kinesia_class"]) == {"off", "on", "dyskinetic"}
    assert features[["W_move_mean", "W_tremor_mean"]].notna().all().all()
    assert features["age"].notna().all() and features["sex"].notna().all()  # demographics joined
    assert features["updrs_off_total"].eq(41).all()

    summary = read_json(reports / "cleaning_summary.json")
    assert summary["kept"] == 16 and summary["total"] == 28
    assert summary["removed"] == {"no_label": 8, "not_worn": 4}

    assert read_json(reports / "failed_participants.json") == []
    assert read_json(reports / "validation_report.json") == []
    assert read_json(reports / "anomalies.json")["critical"] == []
    assert "skipped" not in read_json(reports / "bias_report.json")["model_level"]
    assert (reports / "bias_report.md").read_text(encoding="utf-8").startswith("# Bias report")
    for name in ["schema_baseline.json", "schema_latest.json", "stats_history.csv"]:
        assert (reports / name).exists(), name
    assert len(read_json(processed / "label_issues.json")) > 0  # the "No Data" hours are reported

    # Zips you supplied yourself are never deleted.
    assert sorted(p.name for p in zips.glob("*.zip")) == [f"{pid}.zip" for pid in PIDS]


def test_a_second_run_is_checked_against_the_first(tmp_path, monkeypatch, alerts):
    zips = participants(tmp_path)
    cfg = use_config(monkeypatch, make_config(
        tmp_path / "project", data={"local_zip_dir": str(zips)}, run={"parallel_participants": 1}))
    _, reports = outputs(cfg)

    assert run_pipeline.main([]) == 0
    baseline = (reports / "schema_baseline.json").read_text(encoding="utf-8")
    assert run_pipeline.main([]) == 0

    assert (reports / "schema_baseline.json").read_text(encoding="utf-8") == baseline
    history = pd.read_csv(reports / "stats_history.csv")
    assert len(history) == 2 and history["n_rows"].eq(16).all()
    assert len(list((reports / "stats").glob("stats_*.json"))) >= 1


def test_the_run_stops_when_too_many_participants_fail(tmp_path, monkeypatch, alerts):
    zips = participants(tmp_path, ["COPS-1"])
    with zipfile.ZipFile(zips / "COPS-2.zip", "w") as z:  # no diary inside
        z.writestr("COPS-2/readme.txt", "no diary")
    cfg = use_config(monkeypatch, make_config(
        tmp_path / "project", data={"local_zip_dir": str(zips)}, run={"parallel_participants": 1}))

    assert run_pipeline.main([]) == 1  # 1 of 2 failed, over the 20% limit

    processed, reports = outputs(cfg)
    failed = read_json(reports / "failed_participants.json")
    assert [f["participant_id"] for f in failed] == ["COPS-2"]
    assert alerts == ["[COPS pipeline] 1 of 2 participants failed"]
    assert not (processed / "hourly_features.csv").exists()  # nothing half-finished is published


def test_data_with_no_usable_hours_stops_with_a_critical_alert(tmp_path, monkeypatch, alerts):
    zips = participants(tmp_path, ["COPS-1", "COPS-2"])
    # The production setting: an hour needs most of 3,600 seconds of data. The
    # synthetic hours have 120, so every hour is dropped as low quality.
    cfg = use_config(monkeypatch, make_config(
        tmp_path / "project", data={"local_zip_dir": str(zips)}, run={"parallel_participants": 1},
        features={"expected_seconds_per_file": 3600}))

    with pytest.raises(DataAnomalyError, match="empty_dataset"):
        run_pipeline.main([])

    _, reports = outputs(cfg)
    assert read_json(reports / "cleaning_summary.json")["kept"] == 0
    assert "[COPS pipeline] Critical data anomalies" in alerts
