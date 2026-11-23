"""Signal processing: windows (golden values), wear, orientation and the hourly features."""

import numpy as np
import pandas as pd
import pytest

from data_pipeline import sensor
from tests.helpers import make_config

HOUR = 10 * 3600


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path)


def hour_df(seconds=60, start=0.0, x=None, temp=31.0, fs=100):
    """seconds of samples from 10:00:00 + start, in the shape read.read_hour_csv gives."""
    t = HOUR + start + np.arange(int(round(seconds * fs))) / fs
    return pd.DataFrame({"Time": t, "X": -0.45 if x is None else x(t), "Y": 0.74, "Z": -0.49, "Temp": temp})


def sine(freq, amp=0.1):
    return lambda t: -0.45 + amp * np.sin(2 * np.pi * freq * t)

# ---- windows: golden values ----------------------------------------------------------------


def test_a_minute_gives_six_windows_on_the_diary_clock(cfg):
    w = sensor.hour_windows(hour_df(60), offset_s=86400, cfg=cfg)
    assert w["time_s"].tolist() == [86400 + HOUR + 10 * i for i in range(6)]
    assert w["signal"].shape == (6, 3, 300) and w["signal"].dtype == np.float32
    np.testing.assert_allclose(w["mean"], [[-0.45, 0.74, -0.49]] * 6, atol=1e-6)
    np.testing.assert_allclose(w["signal"], 0, atol=1e-4)  # mean removed (float32 rounding aside)
    np.testing.assert_allclose(w["temp"], 31.0)


def test_a_full_hour_gives_360_windows(cfg):
    assert len(sensor.hour_windows(hour_df(3600), 0, cfg)["time_s"]) == 360


def test_a_real_hour_stamped_at_the_end_of_each_sample_gives_360_windows(cfg):
    # COPS hours run 10:00:00.010 to 10:59:59.990 (11:00:00.000 belongs to the next file's hour).
    w = sensor.hour_windows(hour_df(3600, start=0.01).iloc[:-1], 0, cfg)
    assert len(w["time_s"]) == 360 and w["time_s"][0] == HOUR


def test_windows_start_on_a_10_s_boundary(cfg):
    w = sensor.hour_windows(hour_df(40, start=3.7), 0, cfg)
    assert w["time_s"].tolist() == [HOUR + 10, HOUR + 20, HOUR + 30]  # 3.7 s in: the first whole window starts at 10 s


def test_a_tremor_survives_resampling_and_fast_noise_does_not(cfg):
    def amplitude(freq):  # a sine's amplitude is its RMS x sqrt(2), however it's sampled
        signal = sensor.hour_windows(hour_df(60, x=sine(freq)), 0, cfg)["signal"][1:-1, 0]
        return np.sqrt(2 * np.mean(signal.astype(float) ** 2))
    assert amplitude(5) == pytest.approx(0.1, rel=0.02)  # 5 Hz kept at 30 Hz
    assert amplitude(20) < 0.005  # 20 Hz is above 15 Hz: removed, not aliased


def test_no_window_spans_a_gap(cfg):
    df = hour_df(60)
    df = df[(df["Time"] < HOUR + 20) | (df["Time"] >= HOUR + 25)]
    assert sensor.hour_windows(df, 0, cfg)["time_s"].tolist() == [HOUR + s for s in (0, 10, 30, 40, 50)]


def test_impossible_values_are_clipped(cfg):
    w = sensor.hour_windows(hour_df(30, x=lambda t: np.full_like(t, 50.0)), 0, cfg)
    np.testing.assert_allclose(w["mean"][:, 0], 8.0)


def test_too_little_data_gives_no_windows(cfg):
    w = sensor.hour_windows(hour_df(5), 0, cfg)
    assert w["signal"].shape == (0, 3, 300) and w["time_s"].dtype == np.int64

# ---- wear ----------------------------------------------------------------------------------


def test_a_cold_still_watch_is_not_worn(cfg):
    n = 200
    time_s = np.arange(n) * 10
    still, moving = np.zeros((n, 3, 300), np.float32), np.random.default_rng(0).normal(0, 0.05, (n, 3, 300))
    cold, warm, diary = np.full(n, 20.0), np.full(n, 31.0), np.ones(n, bool)
    assert not sensor.wear_flags(time_s, still, cold, diary, cfg).any()
    assert sensor.wear_flags(time_s, still, warm, diary, cfg).all()   # a resting arm is warm
    assert sensor.wear_flags(time_s, moving, cold, diary, cfg).all()  # cold but moving: worn
    assert not sensor.wear_flags(time_s, moving, warm, ~diary, cfg).any()  # the diary says not worn

# ---- orientation ---------------------------------------------------------------------------


def test_a_watch_turned_round_reads_along_the_arm_above_zero():
    assert sensor.is_turned_round([0.3, 0.5])
    assert not sensor.is_turned_round([-0.4, -0.2])
    assert not sensor.is_turned_round([])


@pytest.mark.parametrize("turned, wrist, expected, mirrored", [
    (False, "right", [1, 1, 1], False),
    (True, "right", [-1, -1, 1], False),
    (False, "left", [1, -1, 1], True),
    (True, "left", [-1, 1, 1], True),
])
def test_orientation_flips(cfg, turned, wrist, expected, mirrored):
    values = np.ones((2, 3, 4))
    assert sensor.orient(values, turned, wrist, cfg) is mirrored
    np.testing.assert_array_equal(values[:, :, 0], [expected] * 2)


def minutes(x, y, z):
    return pd.DataFrame({"X": x, "Y": y, "Z": z}, index=np.arange(len(x)))


@pytest.mark.parametrize("other_turned", [False, True])
def test_mirrored_wrists_agree_once_oriented(cfg, other_turned):
    rng = np.random.default_rng(1)
    x, y, z = -0.5 + rng.normal(0, 0.2, 60), rng.normal(0, 0.3, 60), rng.normal(0, 0.3, 60)
    left = minutes(-x, y, z) if other_turned else minutes(x, -y, z)  # the mirror flips Y; a turn flips X and Y
    result = sensor.wrist_agreement(minutes(x, y, z), left, "left", cfg)
    assert result["minutes"] == 60
    assert min(result["corr_x"], result["corr_y"], result["corr_z"]) == pytest.approx(1.0)


def test_agreement_without_minutes(cfg):
    assert sensor.wrist_agreement(minutes([], [], []), minutes([1], [1], [1]), "right", cfg) == {"minutes": 0}


def test_orientation_can_differ_per_row(cfg):
    values = np.ones((2, 3))
    sensor.orient(values, np.array([True, False]), "right", cfg)
    assert values.tolist() == [[-1, -1, 1], [1, 1, 1]]


def test_each_wear_bout_is_oriented_on_its_own(cfg):
    time_s = np.arange(6 * 360) * 10                 # 6 hours of windows
    worn = np.ones(len(time_s), bool)
    worn[720:1080] = False                           # off for an hour, then put back on turned round
    bouts = sensor.wear_bouts(time_s, worn, 1800)
    assert set(bouts[:720]) == {1} and set(bouts[1080:]) == {2}
    means = np.zeros((len(time_s), 3))
    means[:720, 0], means[1080:, 0] = -0.5, 0.5
    turned = sensor.turned_by_bout(means, worn, np.ones(len(time_s), bool), bouts, cfg)
    assert not turned[:720].any() and turned[1080:].all()


def test_a_short_break_keeps_the_same_bout():
    worn = np.array([True, True, False, True])
    assert sensor.wear_bouts(np.array([0, 10, 20, 30]), worn, 1800).tolist() == [1, 1, 1, 1]


def test_minute_means_are_on_the_diary_clock():
    m = sensor.minute_means(hour_df(120), offset_s=86400)
    assert m.index.tolist() == [(86400 + HOUR) // 60, (86400 + HOUR) // 60 + 1]

# ---- hourly features -----------------------------------------------------------------------


def test_tremor_raises_the_tremor_ratio(cfg):
    tremor = sensor.hour_features(hour_df(120, x=sine(5)), cfg)
    moving = sensor.hour_features(hour_df(120, x=sine(1, 0.3)), cfg)
    assert tremor["status"] == moving["status"] == "ok"
    assert tremor["tremor_ratio"] > 5 * moving["tremor_ratio"]
    assert moving["valid_fraction"] == pytest.approx(1.0)
    assert set(sensor.FEATURE_NAMES) <= set(tremor)


def test_features_need_enough_data(cfg):
    out = sensor.hour_features(hour_df(10), cfg)
    assert out["status"] == "insufficient_data" and np.isnan(out["move_mean"])


def test_features_are_filtered_within_gap_free_runs(cfg):
    df = hour_df(120, x=sine(1, 0.3))
    gapped = df[(df["Time"] < HOUR + 50) | (df["Time"] >= HOUR + 70)]
    out = sensor.hour_features(gapped, cfg)
    assert out["status"] == "ok" and out["valid_fraction"] == pytest.approx(100 / 120)
