import io

import numpy as np
import pandas as pd
import pytest

from scripts.features import (
    ParticipantArchive, bandpass, combine_wrists, compute_hour_features, read_hour_csv, wrist_features,
)
from tests.helpers import csv_bytes, make_config, make_participant_zip, signal


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path)


def test_still_wrist_has_little_movement(cfg):
    f = compute_hour_features(signal("still"), cfg)
    assert f["status"] == "ok"
    assert f["move_mean"] < 0.01
    assert f["still_fraction"] > 0.9


def test_moving_wrist_has_more_movement_than_still(cfg):
    still = compute_hour_features(signal("still"), cfg)
    moving = compute_hour_features(signal("moving"), cfg)
    assert moving["move_mean"] > 5 * still["move_mean"]
    assert moving["still_fraction"] < still["still_fraction"]


def test_tremor_shows_in_tremor_band(cfg):
    tremor = compute_hour_features(signal("tremor"), cfg)
    moving = compute_hour_features(signal("moving"), cfg)
    assert tremor["tremor_ratio"] > moving["tremor_ratio"]
    assert tremor["tremor_mean"] > 5 * moving["tremor_mean"]


def test_bandpass_removes_gravity():
    constant = np.ones(6000)
    assert np.abs(bandpass(constant, 0.1, 3.0, fs=100)).max() < 1e-6


def test_too_little_data_is_flagged(cfg):
    f = compute_hour_features(signal("still", seconds=10), cfg)
    assert f["status"] == "insufficient_data"
    assert np.isnan(f["move_mean"])


def test_missing_values_are_dropped(cfg):
    df = signal("moving")
    df.loc[:500, "X"] = np.nan
    f = compute_hour_features(df, cfg)
    assert f["status"] == "ok" and f["valid_fraction"] < 1


def test_impossible_values_are_clipped(cfg):
    df = signal("still")
    df.loc[0:9, "X"] = 50.0
    assert compute_hour_features(df, cfg)["n_clipped"] == 10


def test_corrupt_inner_zip_does_not_crash(cfg, tmp_path):
    name = "COPS-99_Day0_8h-9h_leftWrist.zip"
    path = make_participant_zip(tmp_path, corrupt=name)
    with ParticipantArchive(path) as archive:
        f = wrist_features(archive, name, None, cfg)
    assert f["status"] == "corrupt_file"


def test_combine_wrists_asymmetry():
    row = combine_wrists({"move_mean": 1.0, "tremor_mean": 0.0, "still_fraction": 0.2},
                         {"move_mean": 3.0, "tremor_mean": 0.0, "still_fraction": 0.1})
    assert row["asym_move_mean"] == pytest.approx(0.5)
    assert np.isnan(row["asym_tremor_mean"])
    assert row["L_move_mean"] == 1.0 and row["R_move_mean"] == 3.0


def test_time_parsing_fast_path_and_fallback():
    import pandas as pd
    from scripts.features import _time_to_seconds
    s = pd.Series(["11:00:00.010", "23:59:59.990", "9:00:00.5", "bad"])
    out = _time_to_seconds(s)
    assert out[0] == pytest.approx(39600.01)
    assert out[1] == pytest.approx(86399.99)
    assert out[2] == pytest.approx(32400.5)  # odd layout uses the fallback
    assert np.isnan(out[3])


# ---------- The fast CSV reader ----------

def _slow_read(data):
    """What read_hour_csv did before the fast reader: pandas, everything as text."""
    return pd.read_csv(io.BytesIO(data), sep=";")


def test_fast_reader_gives_the_same_features_as_pandas(cfg):
    data = csv_bytes(signal("tremor"))
    fast = compute_hour_features(read_hour_csv(io.BytesIO(data)), cfg)
    slow = compute_hour_features(_slow_read(data), cfg)
    assert fast == pytest.approx(slow, nan_ok=True)


def test_fast_reader_turns_clock_times_into_seconds():
    df = read_hour_csv(io.BytesIO(csv_bytes(signal("still", seconds=2))))
    assert pd.api.types.is_float_dtype(df["Time"])
    assert df["Time"].iloc[0] == pytest.approx(11 * 3600 + 0.01)


def test_empty_values_stay_missing_with_the_fast_reader(cfg):
    df = signal("moving")
    df.loc[:500, "X"] = np.nan
    read = read_hour_csv(io.BytesIO(csv_bytes(df)))
    assert read["X"].isna().sum() == 501
    assert compute_hour_features(read, cfg)["valid_fraction"] < 1


def test_a_broken_value_falls_back_to_pandas_and_is_cleaned(cfg):
    df = signal("moving").astype({"X": object})
    df.loc[3, "X"] = "garbage"
    df.loc[4, "Time"] = "not a time"
    read = read_hour_csv(io.BytesIO(csv_bytes(df)))
    assert read["X"].dtype == object  # read as text, by pandas
    f = compute_hour_features(read, cfg)
    assert f["status"] == "ok"
    assert f["valid_fraction"] < 1  # the broken row is dropped


def test_missing_columns_are_still_refused():
    data = csv_bytes(signal("still", seconds=2).drop(columns=["Z"]))
    with pytest.raises(ValueError, match="missing columns"):
        read_hour_csv(io.BytesIO(data))
