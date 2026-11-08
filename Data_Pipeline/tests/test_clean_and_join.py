import pandas as pd
import pytest

from scripts.clean import clean_hours, drop_duplicate_hours
from scripts.join import attach_demographics, merge_participants
from scripts.patient_info import load_demographics, summarise_updrs
from scripts.process_patient import process_participant
from scripts.utils.file_io import read_table
from tests.helpers import make_config, make_demographics, make_participant_zip


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path)


@pytest.fixture
def processed(cfg, tmp_path):
    return process_participant(make_participant_zip(tmp_path), cfg)


def test_process_participant_builds_table(processed):
    table = pd.read_csv(processed["table"])
    assert len(table) == 7
    assert table["updrs_off_total"].iloc[0] == 41
    assert table["levodopa_response"].iloc[0] == pytest.approx((41 - 20) / 41)
    assert {"W_move_mean", "W_tremor_mean"}.issubset(table.columns)
    assert not [c for c in table.columns if c.startswith(("L_", "R_", "asym_"))]


def test_only_the_configured_wrist_is_used(tmp_path):
    # The right wrist's file is corrupt: a left-wrist run never reads it, a right-wrist run does.
    zip_path = make_participant_zip(tmp_path, corrupt="COPS-99_Day0_8h-9h_rightWrist.zip")
    for side, other, status in [("left", "right", "ok"), ("right", "left", "corrupt_file")]:
        cfg = make_config(tmp_path / side, features={"wrist": side})
        table = read_table(process_participant(zip_path, cfg)["table"])
        assert table.loc[table["hour_start"] == 8, "W_status"].iloc[0] == status
        assert f"{side}_zip" in table.columns and f"{other}_zip" not in table.columns


def test_numbered_dosage_columns_process_end_to_end(cfg, tmp_path):
    # Most real participants (e.g. COPS-11) have Dosage_1, Dosage_2 instead of Dosage.
    res = process_participant(make_participant_zip(tmp_path, "COPS-11", dose_columns=2), cfg)
    table = read_table(res["table"])
    assert len(table) == 7
    first = table.loc[table["hour_start"] == 8].iloc[0]
    assert first["dosage_mg"] == "50;50"
    assert first["W_status"] == "ok"


def test_dosage_stays_text_when_merging_layouts(cfg, tmp_path):
    one = process_participant(make_participant_zip(tmp_path, "COPS-1", dose_columns=1), cfg)
    two = process_participant(make_participant_zip(tmp_path, "COPS-2", dose_columns=3), cfg)
    merged, _ = merge_participants([one, two])
    doses = merged.groupby("participant_id")["dosage_mg"].first()
    assert doses.to_dict() == {"COPS-1": "100", "COPS-2": "50;50"}


def test_cleaning_keeps_only_labelled_worn_hours(cfg, processed):
    clean, summary = clean_hours(pd.read_csv(processed["table"]), cfg)
    assert len(clean) == 4  # 4 labelled awake hours with sensor data
    assert set(clean["kinesia_class"]) == {"on", "off", "dyskinetic"}
    assert summary["removed"]["no_label"] == 2
    assert summary["removed"]["not_worn"] == 1


def test_low_valid_fraction_is_removed(cfg, processed):
    df = pd.read_csv(processed["table"])
    df.loc[0, "W_valid_fraction"] = 0.5
    clean, summary = clean_hours(df, cfg)
    assert summary["removed"].get("low_valid_fraction") == 1


def test_missing_updrs_does_not_crash(cfg, tmp_path):
    res = process_participant(make_participant_zip(tmp_path, include_updrs=False), cfg)
    table = pd.read_csv(res["table"])
    assert table["has_updrs_off"].iloc[0] == 0
    assert table["updrs_off_total"].isna().all()


def test_summarise_updrs_with_one_exam():
    out = summarise_updrs({"total": 30, "days_since_exam": 10}, None)
    assert out["updrs_off_total"] == 30 and pd.isna(out["levodopa_response"])


def test_merge_and_demographics(cfg, tmp_path):
    a = process_participant(make_participant_zip(tmp_path, "COPS-1"), cfg)
    b = process_participant(make_participant_zip(tmp_path, "COPS-2"), cfg)
    merged, issues = merge_participants([a, b])
    demo = load_demographics(make_demographics(tmp_path, ["COPS-1", "COPS-2"]), cfg)
    out = attach_demographics(merged, demo)
    assert set(out["participant_id"]) == {"COPS-1", "COPS-2"}
    assert {"sex", "age_group"}.issubset(out.columns)
    assert len(issues) >= 2


def test_duplicates_are_dropped():
    df = pd.DataFrame({"participant_id": ["A", "A"], "hour_index": [1, 1]})
    assert len(drop_duplicate_hours(df)) == 1
