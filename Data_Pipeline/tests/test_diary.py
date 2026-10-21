import numpy as np
import pandas as pd
import pytest

from scripts.diary import DiaryError, map_kinesia_class, parse_diary, reconcile
from tests.helpers import DIARY_COLUMNS, default_hours, make_config


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path)


def raw_diary(pid="COPS-99"):
    return pd.DataFrame([row for row, _ in default_hours(pid)], columns=DIARY_COLUMNS)


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
