import numpy as np
import pytest

from scripts.features import ParticipantArchive, bandpass, combine_wrists, compute_hour_features, wrist_features
from tests.helpers import make_config, make_participant_zip, signal


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
