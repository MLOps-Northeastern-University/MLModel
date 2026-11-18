"""The whole pipeline on synthetic participants: every output, failures, and the command line."""

import numpy as np
import pandas as pd
import pytest

from scripts import acquire, pipeline
from scripts.utils import read_json, read_table
from tests.helpers import make_config, make_demographics, make_local_run, make_participant_zip, raw_dir


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    cfg = make_local_run(tmp_path_factory.mktemp("run"))
    assert pipeline.run(cfg) == 0
    return cfg


def processed(cfg, name):
    return read_table(cfg["resolved_paths"]["processed_dir"] / name)


def test_every_diary_hour_is_kept_with_a_reason_when_unusable(built):
    hours = processed(built, "hours.csv").set_index(["participant_id", "kinesia_text"])
    assert len(hours) == 14  # 7 diary hours x 2 participants
    one = hours.loc["COPS-1"]
    assert pd.isna(one.loc["Sleep", "drop_reason"])  # sleep is a label (asleep)
    assert one.loc["No Data", "drop_reason"] == "no_label"
    assert one.loc["Good Kinesia", "W_status"] == "ok"
    assert (one["drop_reason"] == "not_worn").sum() == 1  # the unworn hour with no files
    assert {"sex", "age_group", "visit", "dbs", "hoehn_yahr", "pd_subtype", "W_tremor_ratio"} <= set(hours.columns)
    assert not {"left_zip", "kinesia_class", "hours_since_dose", "updrs_off_total"} & set(hours.columns)


def test_windows_cover_the_recorded_hours(built):
    folder = built["resolved_paths"]["processed_dir"] / "windows" / "COPS-1"
    time_s, worn, signal = (np.load(folder / f"{n}.npy") for n in ("time_s", "worn", "signal"))
    assert signal.shape == (72, 3, 300) and signal.dtype == np.float32  # 6 recorded 2-min hours x 12 windows
    assert worn.all() and worn.dtype == bool
    assert sorted(set(time_s // 3600)) == [8, 9, 10, 11, 12, 13]


def test_patients_get_their_more_affected_wrist(built):
    patients = processed(built, "patients.csv").set_index("participant_id")
    assert patients["wrist"].to_dict() == {"COPS-1": "left", "COPS-2": "right"}
    assert patients["mirrored"].to_dict() == {"COPS-1": True, "COPS-2": False}
    assert (patients["turned_round_share"] == 0).all() and patients["more_affected_side"].all()
    assert (patients["wear_bouts"] == 6).all()  # 2-min test hours, an hour apart: each is its own bout
    assert patients["dbs"].to_dict() == {"COPS-1": False, "COPS-2": True}


def test_doses_are_events_with_their_ledd(built):
    events = processed(built, "events.csv")
    assert len(events) == 2  # one Levodopa / Carbidopa 100 mg each
    first = events.iloc[0]
    assert (first["kind"], first["ledd"], first["source"]) == ("dose", 100, "on_time")
    assert first["time_s"] == 8 * 3600 + 30 * 60  # the 08:00 hour, at mid-hour


def test_the_dataset_is_described_and_passes_its_checks(built):
    meta = read_json(built["resolved_paths"]["processed_dir"] / "dataset_meta.json")
    assert (meta["spec_version"], meta["seconds"], meta["rate_hz"], meta["axes"]) == (1, 10, 30, 3)
    assert meta["participants"] == 2 and meta["windows"] == 144
    manifest = read_json(built["resolved_paths"]["processed_dir"] / "windows" / "manifest.json")
    assert manifest == {"spec_version": 1, "seconds": 10, "rate_hz": 30, "axes": 3}
    reports = built["resolved_paths"]["reports_dir"]
    assert read_json(reports / "checks.json")["critical"] == []
    assert read_json(reports / "failed_participants.json") == []
    assert (reports / "bias_report.md").exists()
    assert len(read_table(reports / "wrist_agreement.csv")) == 2


def test_the_same_input_gives_the_same_output_in_parallel(tmp_path, built):
    cfg = make_local_run(tmp_path, run={"parallel_participants": 2})
    assert pipeline.run(cfg) == 0
    pd.testing.assert_frame_equal(processed(cfg, "hours.csv"), processed(built, "hours.csv"))


def test_one_bad_participant_is_skipped(tmp_path):
    cfg = make_local_run(tmp_path, pids=("COPS-1", "COPS-2", "COPS-3"), sides=("left", "left", "left"),
                         data={"max_failed_fraction": 0.5})
    (raw_dir(tmp_path) / "COPS-3.zip").write_bytes(b"not a zip")
    assert pipeline.run(cfg) == 0
    failed = read_json(cfg["resolved_paths"]["reports_dir"] / "failed_participants.json")
    assert [f["participant_id"] for f in failed] == ["COPS-3"]
    assert set(processed(cfg, "hours.csv")["participant_id"]) == {"COPS-1", "COPS-2"}


def test_too_many_failures_stop_the_run(tmp_path):
    cfg = make_local_run(tmp_path)
    (raw_dir(tmp_path) / "COPS-2.zip").write_bytes(b"not a zip")
    assert pipeline.run(cfg) == 1


def test_a_participant_missing_from_the_demographics_fails(tmp_path):
    cfg = make_local_run(tmp_path, data={"max_failed_fraction": 0.5, "participants": ["COPS-1", "COPS-2", "COPS-7"]})
    make_participant_zip(raw_dir(tmp_path), "COPS-7")
    assert pipeline.run(cfg) == 0
    failed = read_json(cfg["resolved_paths"]["reports_dir"] / "failed_participants.json")
    assert "no wrist" in failed[0]["error"]


def test_demographics_are_required(tmp_path, monkeypatch):
    cfg = make_local_run(tmp_path)
    (raw_dir(tmp_path) / "Demographics.csv").unlink()  # so OSF is listed, and it isn't there either
    monkeypatch.setattr(acquire, "list_osf_files", lambda cfg, session=None: [{"name": "COPS-1.zip", "size": 1}])
    with pytest.raises(acquire.AcquisitionError, match="Demographics"):
        pipeline.run(cfg)


def test_a_participant_with_one_wrist_recorded_uses_it(tmp_path):
    folder = raw_dir(tmp_path)
    make_participant_zip(folder, "COPS-1", sides=("Left",))
    make_demographics(folder, ["COPS-1"], ["right"])
    assert pipeline.run(make_config(tmp_path, data={"mode": "list", "participants": ["COPS-1"]})) == 0
    patient = read_table(tmp_path / "data" / "processed" / "patients.csv").iloc[0]
    assert (patient["wrist"], patient["more_affected_side"], patient["mirrored"]) == ("left", False, True)


@pytest.mark.parametrize("side, more_affected", [("both", True), ("", False)])
def test_both_or_blank_sides_use_the_fallback_wrist(tmp_path, side, more_affected):
    folder = raw_dir(tmp_path)
    make_participant_zip(folder, "COPS-1")
    make_demographics(folder, ["COPS-1"], [side])
    assert pipeline.run(make_config(tmp_path, data={"mode": "list", "participants": ["COPS-1"]})) == 0
    patient = read_table(tmp_path / "data" / "processed" / "patients.csv").iloc[0]
    assert (patient["wrist"], bool(patient["more_affected_side"])) == ("right", more_affected)


def test_a_fixed_wrist_can_be_chosen(tmp_path):
    cfg = make_local_run(tmp_path, wrist={"choose": "left"})
    assert pipeline.run(cfg) == 0
    patients = processed(cfg, "patients.csv").set_index("participant_id")
    assert set(patients["wrist"]) == {"left"}
    assert patients["more_affected_side"].to_dict() == {"COPS-1": True, "COPS-2": False}


def test_the_dag_keeps_the_participants_that_worked(tmp_path):
    cfg = make_local_run(tmp_path)
    manifest = {"participants": [{"name": "COPS-1.zip"}, {"name": "COPS-2.zip"}]}
    with pytest.raises(pipeline.TooManyFailures):
        pipeline.check_failures(cfg, manifest, [{"participant_id": "COPS-1", "dir": "x"}, None])
    ok = [{"participant_id": "COPS-1", "dir": "x"}, {"participant_id": "COPS-2", "dir": "y"}]
    assert pipeline.check_failures(cfg, manifest, ok) == ok


def test_downloaded_zips_are_deleted_and_local_ones_kept(tmp_path, monkeypatch):
    cfg = make_local_run(tmp_path, data={"delete_raw_after_processing": True})
    patients = pipeline.patients_step(cfg, acquire.fetch_manifest(cfg))
    downloaded = make_participant_zip(cfg["resolved_paths"]["raw_dir"], "COPS-1")
    monkeypatch.setattr(acquire, "download_file", lambda entry, cfg: entry.get("local_path") or downloaded)
    pipeline.participant_step(cfg, {"name": "COPS-1.zip"}, patients)
    assert not downloaded.exists()
    local = raw_dir(tmp_path) / "COPS-2.zip"
    pipeline.participant_step(cfg, {"name": "COPS-2.zip", "local_path": str(local)}, patients)
    assert local.exists()


def test_the_command_line(monkeypatch):
    seen = {}
    monkeypatch.setattr(pipeline, "run", lambda cfg, mode: seen.update(cfg=cfg, mode=mode) or 0)
    assert pipeline.main(["--mode", "full"]) == 0
    assert seen["mode"] == "full" and seen["cfg"]["data"]["mode"] == "full"
