from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model_pipeline.core.errors import ConfigError, SchemaError
from model_pipeline.core.targets import (
    ASLEEP, AWAKE, KINESIA_SCORES, LABEL_NAMES, SLEEP_CLASSES, STATE_CLASSES, STATE_GROUPS, TARGETS,
    TREMOR_CLASSES, TREMOR_SCORES, at_horizon, build_labels, get_target, label_name)

# State class indices for the fixture's answers: index = score + 3.
SLIGHT_AKINESIA, GOOD, SEVERE_DYSKINESIA = 2, 3, 6
NAN = np.nan


@pytest.fixture
def table():
    # P1 has hours 10-14 in a row; P2 has hour 10 only, so its next hour is missing.
    return pd.DataFrame({
        "participant_id": ["P1", "P1", "P1", "P1", "P1", "P2"],
        "hour_index": [10, 11, 12, 13, 14, 10],
        "kinesia_text": ["Slight Akinesia", "Good Kinesia", "Severe Dyskinesia", "Sleep", "No Data",
                         "Good Kinesia"],
        "kinesia_score": [-1, 0, 3, NAN, NAN, 0],
        "tremor_score": [2, 0, 1, 0, NAN, 1],
    })


def assert_values(series, expected):
    np.testing.assert_array_equal(series.to_numpy(), np.array(expected, dtype=float))


def test_state_is_the_kinesia_score_and_sleep_its_own_target(table):
    assert_values(get_target("state").labels(table),
                  [SLIGHT_AKINESIA, GOOD, SEVERE_DYSKINESIA, NAN, NAN, GOOD])   # no waking state while asleep
    assert_values(get_target("sleep").labels(table), [AWAKE, AWAKE, AWAKE, ASLEEP, NAN, AWAKE])
    assert get_target("state").classes == STATE_CLASSES and get_target("sleep").classes == SLEEP_CLASSES


def test_classes_line_up_with_scores_and_groups():
    assert [STATE_CLASSES[s + 3] for s in (-3, 0, 3)] == ["severe_akinesia", "good_kinesia", "severe_dyskinesia"]
    assert len(KINESIA_SCORES) == len(STATE_CLASSES) == 7 and SLEEP_CLASSES[ASLEEP] == "asleep"
    assert len(TREMOR_SCORES) == len(TREMOR_CLASSES)
    # Every class belongs to exactly one broad state, and off is the akinesia scores.
    assert sorted(i for group in STATE_GROUPS.values() for i in group) == list(range(len(STATE_CLASSES)))
    assert [KINESIA_SCORES[i] for i in STATE_GROUPS["off"]] == [-3, -2, -1]


def test_off_does_not_depend_on_tremor(table):
    no_tremor = table.assign(tremor_score=0)
    pd.testing.assert_series_equal(get_target("state").labels(table),
                                   get_target("state").labels(no_tremor))


def test_tremor_thresholds(table):
    assert_values(get_target("tremor").labels(table), [2, 0, 1, NAN, NAN, 1])


def test_sleep_hours_are_asleep_with_no_state_or_tremor_label(table):
    sleep = table.index[table["kinesia_text"] == "Sleep"][0]
    assert get_target("sleep").labels(table)[sleep] == ASLEEP
    assert np.isnan(get_target("state").labels(table)[sleep]) and np.isnan(get_target("tremor").labels(table)[sleep])


def test_no_data_is_nan_not_zero(table):
    no_data = table.index[table["kinesia_text"] == "No Data"][0]
    for name in TARGETS:
        assert np.isnan(get_target(name).labels(table)[no_data])


def test_next_hour_stays_within_a_participant(table):
    y = get_target("state").labels(table)
    # P1's last hour has no next hour, and P2's hour 10 must not borrow P1's hour 11.
    assert_values(at_horizon(table, y, "next_hour"), [GOOD, SEVERE_DYSKINESIA, NAN, NAN, NAN, NAN])
    sleep = get_target("sleep").labels(table)
    assert_values(at_horizon(table, sleep, "next_hour"), [AWAKE, AWAKE, ASLEEP, NAN, NAN, NAN])
    assert at_horizon(table, y, "now") is y


def test_next_hour_skips_gaps(table):
    gap = table.drop(index=1)   # P1's hour 11 missing
    y = get_target("state").labels(gap)
    assert np.isnan(at_horizon(gap, y, "next_hour")[0])


def test_build_labels_names_and_order(table):
    labels = build_labels(table, ["sleep", "state", "tremor"])
    assert list(labels.columns) == ["sleep_now", "sleep_next_hour", "state_now", "state_next_hour",
                                    "tremor_now", "tremor_next_hour"]
    assert list(labels.index) == list(table.index)
    assert list(build_labels(table, ["state"], horizons=["now"]).columns) == ["state_now"]


def test_label_names():
    assert label_name("state", "now") == "state_now"
    assert LABEL_NAMES == {f"{t}_{h}" for t in ("sleep", "state", "tremor") for h in ("now", "next_hour")}


def test_labels_keep_index_and_name(table):
    table.index = [10, 11, 12, 13, 14, 15]
    y = get_target("state").labels(table)
    assert list(y.index) == list(table.index) and y.name == "state"


@pytest.mark.parametrize("change, match", [
    ({"kinesia_score": [-4, 0, 3, NAN, NAN, 0]}, "-3 to 3"),
    ({"kinesia_score": [-1, 0, 2.5, NAN, NAN, 0]}, "-3 to 3"),
    ({"tremor_score": ["abc", 0, 1, 0, NAN, 1]}, "isn't a number"),
    ({"tremor_score": [3, 0, 1, 0, NAN, 1]}, "0 to 2"),
])
def test_bad_values_raise(table, change, match):
    with pytest.raises(SchemaError, match=match):
        build_labels(table.assign(**change), list(TARGETS))


def test_missing_column_raises(table):
    with pytest.raises(SchemaError, match="kinesia_text"):
        get_target("state").labels(table.drop(columns="kinesia_text"))
    with pytest.raises(SchemaError, match="hour_index"):
        build_labels(table.drop(columns="hour_index"), ["state"])


def test_duplicate_hour_raises(table):
    twice = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaError, match="twice"):
        build_labels(twice, ["state"])


def test_unknown_names_raise(table):
    with pytest.raises(ConfigError, match="unknown target"):
        get_target("freezing")
    with pytest.raises(ConfigError, match="unknown horizon"):
        build_labels(table, ["state"], horizons=["tomorrow"])
    with pytest.raises(ConfigError):
        build_labels(table, [])


def test_on_real_table_if_present():
    # hourly_all.csv keeps sleep and no-data hours; hourly_features.csv drops them.
    path = Path(__file__).resolve().parents[3] / "Data_Pipeline/data/processed/hourly_all.csv"
    if not path.exists():
        pytest.skip("data pipeline output not pulled")
    table = pd.read_csv(path, low_memory=False)
    labels = build_labels(table, list(TARGETS))
    assert (labels["sleep_now"] == ASLEEP).sum() == (table["kinesia_text"] == "Sleep").sum()
    assert labels["state_now"].notna().sum() == table["kinesia_score"].notna().sum()
    asleep = (labels["sleep_now"] == ASLEEP).sum()
    assert labels["sleep_now"].notna().sum() == labels["state_now"].notna().sum() + asleep
    # The data pipeline's off class is the same hours as the score's off group.
    off = labels["state_now"].isin(STATE_GROUPS["off"])
    assert (off == (table["kinesia_class"] == "off")).all()
    assert labels["tremor_now"].notna().sum() <= table["tremor_score"].notna().sum()
