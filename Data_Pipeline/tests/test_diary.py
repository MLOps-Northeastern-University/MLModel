"""The diary: one row per hour, nothing dropped, the text trusted over the score."""

import io

import numpy as np
import pandas as pd
import pytest

from scripts.diary import DiaryError, dosage_columns, parse_diary, reconcile
from tests.helpers import csv_bytes, default_hours, diary_columns, diary_row, make_config


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path)


def raw(rows, dose_columns=1):
    return pd.read_csv(io.BytesIO(csv_bytes(pd.DataFrame(rows, columns=diary_columns(dose_columns)))), sep=";")


@pytest.fixture
def hours(cfg):
    return parse_diary(raw([r for r, _ in default_hours("COPS-1")]), cfg)


def test_every_hour_is_kept(hours):
    table, _ = hours
    assert len(table) == 7
    assert table["kinesia_text"].tolist()[4:6] == ["Sleep", "No Data"]


def test_the_hour_convention(cfg, hours):
    first = hours[0].iloc[0]  # Day "0 days", Time "9 hr" = 08:00-09:00
    assert (first["day"], first["hour_start"], first["hour_index"]) == (0, 8, 8)
    later = parse_diary(raw([diary_row("COPS-1", 2, 18, "Good Kinesia", 0)]), cfg)[0].iloc[0]
    assert later["hour_index"] == 2 * 24 + 17


def test_sleep_and_no_data_have_no_score(hours):
    table = hours[0].set_index("kinesia_text")
    assert np.isnan(table.loc["Sleep", "kinesia_score"]) and np.isnan(table.loc["No Data", "kinesia_score"])
    assert table.loc["Slight Akinesia", "kinesia_score"] == -1


def test_a_score_without_a_label_is_reported(hours):
    issues = pd.DataFrame(hours[1])
    assert set(zip(issues["field"], issues["issue"])) == {("kinesia", "score_without_label")}


def test_reconcile_trusts_the_text():
    mapping = {"Slight Akinesia": -1}
    assert reconcile("Slight Akinesia", -1, mapping) == (-1, None)
    assert reconcile("Slight Akinesia", 2, mapping) == (-1, "score_text_mismatch")
    assert reconcile("Slight Akinesia", np.nan, mapping) == (-1, "score_missing")
    assert reconcile("Sleep", np.nan, mapping)[1] is None


def test_text_is_stripped(cfg):
    table, _ = parse_diary(raw([diary_row("COPS-1", 0, 3, " Sleep ", None)]), cfg)
    assert table["kinesia_text"][0] == "Sleep"


@pytest.mark.parametrize("dose_columns", [1, 2, 4])
def test_every_dosage_layout_is_read(cfg, dose_columns):
    rows = [r for r, _ in default_hours("COPS-1", dose_columns)]
    table, _ = parse_diary(raw(rows, dose_columns), cfg)
    assert table["dosage_mg"][0] == ("100" if dose_columns == 1 else "50;50")


def test_dosage_columns_are_ordered_by_number():
    assert dosage_columns(["Dosage_10", "x", "Dosage_2", "Dosage_1"]) == ["Dosage_1", "Dosage_2", "Dosage_10"]


def test_a_repeated_hour_is_kept_once(cfg):
    row = diary_row("COPS-1", 0, 3, "Good Kinesia", 0)
    assert len(parse_diary(raw([row, row]), cfg)[0]) == 1


def test_bad_diaries_are_refused(cfg):
    with pytest.raises(DiaryError, match="missing columns"):
        parse_diary(pd.DataFrame({"ID": ["COPS-1"]}), cfg)
    with pytest.raises(DiaryError, match="no rows"):
        parse_diary(raw([]), cfg)
    bad = diary_row("COPS-1", 0, 3, "Good Kinesia", 0)
    bad["Day"] = "unknown"
    with pytest.raises(DiaryError, match="isn't a number"):
        parse_diary(raw([bad]), cfg)
