import io

import numpy as np
import pandas as pd
import pytest

from scripts.diary import DiaryError, dosage_columns, load_diary, map_kinesia_class, parse_diary, reconcile
from tests.helpers import csv_bytes, default_hours, diary_columns, make_config


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path)


def raw_diary(pid="COPS-99", dose_columns=1):
    rows = [row for row, _ in default_hours(pid, dose_columns)]
    return pd.DataFrame(rows, columns=diary_columns(dose_columns))


def test_hour_convention(cfg):
    diary, _ = parse_diary(raw_diary(), cfg)
    first = diary.iloc[0]
    assert first["day"] == 0 and first["hour_start"] == 8  # "9 hr" = 08:00-09:00
    assert first["left_zip"].endswith("_8h-9h_leftWrist.zip")


@pytest.mark.parametrize("score,expected", [(-3, "off"), (-1, "off"), (0, "on"), (1, "dyskinetic"), (3, "dyskinetic")])
def test_kinesia_classes(cfg, score, expected):
    assert map_kinesia_class(score, cfg["labels"]["kinesia_classes"]) == expected


def test_sleep_and_no_data_have_no_label(cfg):
    diary, _ = parse_diary(raw_diary(), cfg)
    by_text = diary.set_index("kinesia_text")
    assert pd.isna(by_text.loc["Sleep", "kinesia_class"])
    assert pd.isna(by_text.loc["No Data", "kinesia_score"])


def test_text_score_mismatch_is_reported(cfg):
    _, issues = parse_diary(raw_diary(), cfg)
    assert any(i["issue"] == "score_without_label" and i["field"] == "kinesia" for i in issues)


def test_reconcile_trusts_text():
    mapping = {"Good Kinesia": 0}
    assert reconcile("Good Kinesia", 2, mapping) == (0, "score_text_mismatch")
    assert reconcile("Good Kinesia", np.nan, mapping) == (0, "score_missing")
    assert reconcile("Good Kinesia", 0, mapping) == (0, None)


def test_hours_since_dose(cfg):
    diary, _ = parse_diary(raw_diary(), cfg)
    assert diary["hours_since_dose"].tolist()[:3] == [0, 1, 2]


def test_missing_columns_raise(cfg):
    with pytest.raises(DiaryError):
        parse_diary(raw_diary().drop(columns=["KinesiaScore"]), cfg)


# The real diaries come in 4 layouts: "Dosage" (21 participants) or
# "Dosage_1".."Dosage_N" with N = 2, 3 or 4 (45 participants).
@pytest.mark.parametrize("dose_columns", [1, 2, 3, 4])
def test_every_dosage_layout_parses(cfg, dose_columns):
    # Round-trip through CSV text, as the pipeline reads it from the zip.
    raw = load_diary(io.BytesIO(csv_bytes(raw_diary(dose_columns=dose_columns))))
    diary, _ = parse_diary(raw, cfg)
    first = diary.iloc[0]
    if dose_columns == 1:
        assert first["dosage_mg"] == "100"
    else:
        assert first["medication"] == "Levodopa / Benserazid;Opicapone"
        assert first["dosage_mg"] == "50;50"
    assert first["medication_taken"] == 1
    assert diary["dosage_mg"].iloc[1:].isna().all()


def test_dosage_columns_are_ordered_by_number():
    columns = ["ID", "Dosage_2", "Medication", "Dosage_10", "Dosage_1", "Dosages"]
    assert dosage_columns(columns) == ["Dosage_1", "Dosage_2", "Dosage_10"]
    assert dosage_columns(["Dosage", "Medication"]) == ["Dosage"]


def test_diary_without_dosage_column_still_parses(cfg):
    diary, _ = parse_diary(raw_diary().drop(columns=["Dosage"]), cfg)
    assert diary["dosage_mg"].isna().all()
    assert diary["medication_taken"].iloc[0] == 1
