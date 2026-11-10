import numpy as np
import pandas as pd
import pytest

from scripts.core.errors import ConfigError, SchemaError
from scripts.core.targets import BINARY, ORDINAL, TARGETS, build_labels, get_target


@pytest.fixture
def table():
    # off, on, dyskinetic, asleep/no data
    return pd.DataFrame({
        "kinesia_class": ["off", "on", "dyskinetic", None],
        "tremor_score": [2, 0, 1, np.nan],
    })


def test_off_is_akinesia_only(table):
    assert get_target("off").labels(table).tolist()[:3] == [1.0, 0.0, 0.0]


def test_dyskinesia_label(table):
    assert get_target("dyskinesia").labels(table).tolist()[:3] == [0.0, 0.0, 1.0]


def test_no_diary_answer_is_nan_not_zero(table):
    for name in TARGETS:
        assert np.isnan(get_target(name).labels(table).iloc[3])


def test_off_does_not_depend_on_tremor(table):
    no_tremor = table.assign(tremor_score=0)
    pd.testing.assert_series_equal(get_target("off").labels(table),
                                   get_target("off").labels(no_tremor))


def test_tremor_is_the_diary_score(table):
    assert get_target("tremor").labels(table).tolist()[:3] == [2.0, 0.0, 1.0]


def test_kinds_and_intervals():
    assert get_target("off").kind == BINARY and not get_target("off").has_interval
    assert get_target("tremor").kind == ORDINAL and get_target("tremor").has_interval


def test_labels_keep_index_and_name(table):
    table.index = [10, 11, 12, 13]
    y = get_target("off").labels(table)
    assert list(y.index) == [10, 11, 12, 13] and y.name == "off"


def test_missing_column_raises():
    with pytest.raises(SchemaError, match="kinesia_class"):
        get_target("off").labels(pd.DataFrame({"tremor_score": [1]}))


def test_out_of_range_raises():
    with pytest.raises(SchemaError, match="outside"):
        get_target("tremor").labels(pd.DataFrame({"tremor_score": [3]}))


def test_unknown_target_raises():
    with pytest.raises(ConfigError, match="unknown target"):
        get_target("freezing")


def test_build_labels(table):
    labels = build_labels(table, ["off", "tremor"])
    assert list(labels.columns) == ["off", "tremor"] and len(labels) == 4
    with pytest.raises(ConfigError):
        build_labels(table, [])


def test_on_real_table_if_present():
    from pathlib import Path
    path = Path(__file__).resolve().parents[3] / "Data_Pipeline/data/processed/hourly_features.csv"
    if not path.exists():
        pytest.skip("data pipeline output not pulled")
    labels = build_labels(pd.read_csv(path, low_memory=False), list(TARGETS))
    assert labels["off"].notna().any()
