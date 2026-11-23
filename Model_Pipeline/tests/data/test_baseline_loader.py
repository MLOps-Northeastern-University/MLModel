import numpy as np
import pandas as pd
import pytest

from scripts.data.baseline_loader import FEATURE_COLUMNS, FEATURE_NAMES, build_features, cohort_stats, settings

B = settings({"warmup_hours": 2})


def hours_table(n=10, pid="COPS-1", seed=0):
    rng = np.random.default_rng(seed)
    h = pd.DataFrame({"participant_id": pid, "day": 0, "hour_start": np.arange(n) % 24, "hour_index": np.arange(n),
                      "kinesia_text": "Good Kinesia", "kinesia_score": 0.0, "tremor_score": 0.0, "drop_reason": np.nan,
                      "W_status": "ok", "W_valid_fraction": 1.0, "worn_fraction": 1.0})
    for name in FEATURE_NAMES:
        h[f"W_{name}"] = rng.uniform(0.01, 0.5, n)
    return h


def doses(*rows):
    return pd.DataFrame([{"participant_id": p, "time_s": t, "kind": "dose", "ledd": ledd} for p, t, ledd in rows])


def test_columns_and_one_row_per_hour():
    t = build_features(hours_table(), doses(("COPS-1", 1800, 100.0)), B)
    assert len(t) == 10 and set(FEATURE_COLUMNS) <= set(t.columns)
    assert t[FEATURE_COLUMNS].notna().all().all()


def test_the_first_hour_is_judged_against_the_cohort():
    h = hours_table()
    cohort = (np.zeros(7), np.ones(7))
    t = build_features(h, doses(), settings({"warmup_hours": 2, "log_features": []}), cohort)
    assert t.loc[0, "z_move_mean"] == pytest.approx(h.loc[0, "W_move_mean"])   # (x - 0) / 1


def test_z_score_uses_only_earlier_hours():
    h = hours_table()
    before = build_features(h, doses(), B, cohort_stats(h, B))
    h.loc[5, "W_move_mean"] = 100.0   # an hour can't change its own or earlier z-scores' reference
    after = build_features(h, doses(), B, cohort_stats(hours_table(), B))
    pd.testing.assert_series_equal(before["z_move_mean"].iloc[:5], after["z_move_mean"].iloc[:5])
    assert after.loc[5, "z_move_mean"] > 5 and after.loc[6, "z_move_mean"] != before.loc[6, "z_move_mean"]


def test_bad_sensor_hours_are_neutral_and_flagged():
    h = hours_table()
    h.loc[5, "W_status"] = "no_file"
    t = build_features(h, doses(), B)
    assert t.loc[5, "z_move_mean"] == 0 and t.loc[5, "sensor_ok"] == 0 and t.loc[4, "sensor_ok"] == 1


def test_dose_timing_is_at_the_end_of_each_hour():
    # Two doses at 02:30 (summed) and one at 07:30.
    given = doses(("COPS-1", 9000, 100.0), ("COPS-1", 9000, 50.0), ("COPS-1", 27000, 75.0))
    t = build_features(hours_table(), given, B)
    assert t.loc[1, "no_dose_yet"] == 1 and t.loc[1, "hours_since_dose"] == 12
    assert t.loc[2, "hours_since_dose"] == 0.5 and t.loc[2, "last_dose_ledd"] == 150
    assert t.loc[6, "ledd_recent"] == 0          # 02:30 is more than 4 h before 07:00
    assert t.loc[7, "ledd_recent"] == 75 and t.loc[5, "ledd_recent"] == 150


def test_doses_stay_with_their_patient():
    h = pd.concat([hours_table(pid="COPS-1"), hours_table(pid="COPS-2")], ignore_index=True)
    t = build_features(h, doses(("COPS-2", 1800, 100.0)), B)
    assert (t.loc[t.participant_id == "COPS-1", "no_dose_yet"] == 1).all()
    assert t.loc[t.participant_id == "COPS-2", "no_dose_yet"].iloc[0] == 0


def test_time_of_day_is_a_circle():
    t = build_features(hours_table(24), doses(), B)
    gap = np.hypot(t.loc[0, "tod_sin"] - t.loc[23, "tod_sin"], t.loc[0, "tod_cos"] - t.loc[23, "tod_cos"])
    assert gap == pytest.approx(2 * np.sin(np.pi / 24))   # 23:00 and 00:00 are one hour apart
    assert np.allclose(t["tod_sin"] ** 2 + t["tod_cos"] ** 2, 1)
