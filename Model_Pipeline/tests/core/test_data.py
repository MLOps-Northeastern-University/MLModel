from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from model_pipeline.core.data import GROUPS, WindowSpec
from model_pipeline.core.errors import ConfigError, SchemaError
from model_pipeline.utils.config import load_config
from tests.helpers import make_dataset

SPEC = WindowSpec(version=1, seconds=10, rate_hz=30, axes=3)


@pytest.fixture
def data():
    return make_dataset(SPEC)


def with_events(data, **changes):
    """The dataset with the first dose (row 0) or first check-in (row 1) changed."""
    events = data.events.copy()
    for column, (row, value) in changes.items():
        events.loc[row, column] = value
    return replace(data, events=events)


# ---- the window spec -------------------------------------------------------------------

def test_spec_comes_from_the_config():
    spec = WindowSpec.from_config(load_config())
    assert spec == SPEC and spec.samples == 300


@pytest.mark.parametrize("windows", [None, {"spec_version": 1, "seconds": 10, "rate_hz": 30},
                                     {"spec_version": 1, "seconds": 0, "rate_hz": 30, "axes": 3},
                                     {"spec_version": 1, "seconds": "10", "rate_hz": 30, "axes": 3},
                                     {"spec_version": True, "seconds": 10, "rate_hz": 30, "axes": 3}])
def test_bad_spec_config_raises(windows):
    with pytest.raises(ConfigError, match="windows"):
        WindowSpec.from_config({} if windows is None else {"windows": windows})


# ---- the dataset ------------------------------------------------------------------------

def test_a_complete_dataset_passes(data):
    data.check(GROUPS, SPEC)
    assert data.participants == ("COPS-1", "COPS-2")


def test_windows_line_up_with_diary_hours(data):
    hours = data.windows["COPS-1"].hour_index()
    assert len(hours) == 3 * 360
    assert sorted(set(hours)) == sorted(data.hours.loc[data.hours.participant_id == "COPS-1", "hour_index"])


def test_subset_keeps_only_those_participants_in_every_group(data):
    one = data.subset(["COPS-2"])
    one.check(GROUPS, SPEC)
    assert one.participants == ("COPS-2",)
    assert set(one.windows) == {"COPS-2"}
    assert len(one.labels) == len(one.features) == len(one.hours) == 3
    assert set(one.events.participant_id) == set(one.patients.participant_id) == {"COPS-2"}


def test_subset_of_an_unknown_participant_raises(data):
    with pytest.raises(SchemaError, match="not in this dataset"):
        data.subset(["COPS-9"])


def test_only_the_groups_a_model_needs_are_required(data):
    replace(data, windows=None, events=None).check(("features",), SPEC)
    with pytest.raises(SchemaError, match="windows group"):
        replace(data, windows=None).check(("windows",), SPEC)
    with pytest.raises(ConfigError, match="unknown input group"):
        data.check(("medication",), SPEC)


def test_windows_cut_with_another_spec_are_refused(data):
    with pytest.raises(SchemaError, match="cut with spec v1"):
        data.check(("windows",), replace(SPEC, version=2))
    # The features don't depend on how windows are cut.
    data.check(("features",), replace(SPEC, version=2))


@pytest.mark.parametrize("change, match", [
    (lambda d: replace(d, hours=pd.concat([d.hours, d.hours.iloc[[0]]])), "twice"),
    (lambda d: replace(d, labels=d.labels.iloc[1:]), "one row per hour"),
    (lambda d: replace(d, labels=d.labels.rename(columns={"state_now": "freezing_now"})), "unknown label"),
    (lambda d: replace(d, hours=d.hours.drop(columns="hour_index")), "hour_index"),
])
def test_hours_and_labels_must_agree(data, change, match):
    with pytest.raises(SchemaError, match=match):
        change(data)


# ---- each group's schema ----------------------------------------------------------------

def bad_windows(data, **arrays):
    w = data.windows["COPS-1"]
    return replace(data, windows={**data.windows, "COPS-1": replace(w, **arrays)})


@pytest.mark.parametrize("arrays, match", [
    (lambda w: {"signal": w.signal[:, :, :240]}, "spec v1"),
    (lambda w: {"signal": w.signal.astype(np.float64)}, "float32"),
    (lambda w: {"worn": w.worn.astype(int)}, "booleans"),
    (lambda w: {"time_s": w.time_s - 5 * np.arange(len(w))}, "increase by at least 10 s"),
])
def test_bad_windows_raise(data, arrays, match):
    with pytest.raises(SchemaError, match=match):
        bad_windows(data, **arrays(data.windows["COPS-1"])).check(("windows",), SPEC)


def test_every_participant_needs_windows(data):
    with pytest.raises(SchemaError, match="have none"):
        replace(data, windows={"COPS-1": data.windows["COPS-1"]}).check(("windows",), SPEC)


@pytest.mark.parametrize("changes, match", [
    ({"kind": (0, "meal")}, "events.kind"),
    ({"ledd": (0, -5.0)}, "LEDD"),
    ({"ledd": (0, np.nan)}, "LEDD"),
    ({"source": (0, "guessed")}, "events.source"),
    ({"time_s": (0, np.nan)}, "time_s"),
    ({"participant_id": (0, "COPS-9")}, "not in hours"),
    ({"kinesia_score": (1, np.nan), "tremor_score": (1, np.nan)}, "neither answer"),
    ({"kinesia_score": (1, 4.0)}, "events.kinesia_score"),
    ({"tremor_score": (1, 3.0)}, "events.tremor_score"),
])
def test_bad_events_raise(data, changes, match):
    with pytest.raises(SchemaError, match=match):
        with_events(data, **changes).check(("events",), SPEC)


def test_a_check_in_with_one_answer_is_fine(data):
    with_events(data, tremor_score=(1, np.nan)).check(("events",), SPEC)


@pytest.mark.parametrize("patients, match", [
    (lambda p: pd.concat([p, p.iloc[[0]]]), "more than one row"),
    (lambda p: p.iloc[1:], "have no row"),
    (lambda p: p.assign(dbs=["yes", "no"]), "patients.dbs"),
    (lambda p: p.drop(columns="more_affected_side"), "more_affected_side"),
])
def test_bad_patients_raise(data, patients, match):
    with pytest.raises(SchemaError, match=match):
        replace(data, patients=patients(data.patients)).check(("patients",), SPEC)


def test_features_must_be_numbers_on_the_hours(data):
    with pytest.raises(SchemaError, match="numeric"):
        replace(data, features=data.features.assign(move_mean="high")).check(("features",), SPEC)
    with pytest.raises(SchemaError, match="one row per hour"):
        replace(data, features=data.features.iloc[1:]).check(("features",), SPEC)
