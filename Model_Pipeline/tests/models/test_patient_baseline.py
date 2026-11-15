import numpy as np
import pytest

from scripts.core.errors import ConfigError, ModelError
from scripts.models import patient_baseline as pb

H = 3600
WINDOW_S = 360.0          # 10 windows an hour, so warm-up weights stay readable


def recording(n_hours=30, per_hour=10, features=4, seed=0):
    rng = np.random.default_rng(seed)
    t = np.concatenate([h * H + np.arange(per_hour) * WINDOW_S for h in range(n_hours)])
    x = rng.normal(size=(len(t), features)) + np.repeat(np.arange(n_hours), per_hour)[:, None] * 0.1
    return t, x.astype(np.float32)


def test_settings_fill_defaults_and_refuse_nonsense():
    assert pb.check_settings(None) == pb.DEFAULTS
    assert pb.check_settings({"mode": "rolling"})["window_h"] == 168
    for bad in ({"mode": "daily"}, {"window_h": 0}, {"warmup_h": -1}, {"window_h": 2.5}, {"colour": 1}):
        with pytest.raises(ConfigError, match="baseline"):
            pb.check_settings(bad)


def test_feature_stats_match_numpy_over_chunks():
    _, x = recording()
    mean, sd = pb.feature_stats([x[:37], x[37:200], x[200:]])
    np.testing.assert_allclose(mean, x.astype(np.float64).mean(0), rtol=1e-10)
    np.testing.assert_allclose(sd, x.astype(np.float64).std(0, ddof=1), rtol=1e-10)
    with pytest.raises(ModelError, match="at least 2"):
        pb.feature_stats([x[:1]])


def test_each_hour_is_relative_to_the_past_window_only():
    t, x = recording()
    s = pb.check_settings({"mode": "rolling", "window_h": 5, "warmup_h": 0})
    b = pb.rolling_baseline(t, x, pb.feature_stats([x]), s, WINDOW_S)
    hour = (t // H).astype(int)
    h = 12
    past = x[(hour >= h - 5) & (hour < h)].astype(np.float64)
    np.testing.assert_allclose(b.mean[h], past.mean(0), rtol=1e-5)
    np.testing.assert_allclose(1 / b.inv_sd[h], past.std(0), rtol=1e-4)
    rows = hour == h
    np.testing.assert_allclose(b.apply(t[rows], x[rows]), (x[rows] - past.mean(0)) / past.std(0), rtol=1e-4, atol=1e-5)


def test_the_future_never_changes_an_hour():
    t, x = recording()
    s = pb.check_settings({"mode": "rolling", "window_h": 8, "warmup_h": 2})
    cohort = pb.feature_stats([x])
    before = pb.rolling_baseline(t, x, cohort, s, WINDOW_S)
    changed = x.copy()
    changed[(t // H) >= 20] += 100.0                  # hours 20 onwards move wildly
    after = pb.rolling_baseline(t, changed, cohort, s, WINDOW_S)
    np.testing.assert_array_equal(before.mean[:21], after.mean[:21])   # hour 20 reads only hours 12-19
    np.testing.assert_array_equal(before.inv_sd[:21], after.inv_sd[:21])
    assert not np.allclose(before.mean[25], after.mean[25])


def test_with_no_past_an_hour_uses_the_cohort_and_the_warm_up_blends_it_in():
    t, x = recording()
    cohort = (np.full(4, 5.0), np.full(4, 2.0))
    b = pb.rolling_baseline(t, x, cohort, pb.check_settings({"mode": "rolling", "warmup_h": 3}), WINDOW_S)
    np.testing.assert_allclose(b.mean[0], 5.0)                          # the first hour has no past
    np.testing.assert_allclose(1 / b.inv_sd[0], 2.0)
    own = x[(t // H) == 0].astype(np.float64).mean(0)                   # hour 1: 10 own windows, weight of 30
    np.testing.assert_allclose(b.mean[1], (10 * own + 30 * 5.0) / 40, rtol=1e-5)
    none = pb.rolling_baseline(t, x, cohort, pb.check_settings({"mode": "rolling", "warmup_h": 0}), WINDOW_S)
    np.testing.assert_allclose(none.mean[0], 5.0)                       # nothing at all: still the cohort


def test_a_feature_that_never_varies_is_only_centred():
    t, x = recording()
    x[:, 2] = 7.0
    s = pb.check_settings({"mode": "rolling", "warmup_h": 0})
    b = pb.rolling_baseline(t, x, pb.feature_stats([x]), s, WINDOW_S)
    out = b.apply(t, x)
    assert np.isfinite(out).all() and np.abs(out[:, 2]).max() < 1e-4 and np.all(b.inv_sd[1:, 2] == 1.0)


def test_normalised_reads_slices_and_rows_like_apply():
    t, x = recording()
    s = pb.check_settings({"mode": "rolling"})
    b = pb.rolling_baseline(t, x, pb.feature_stats([x]), s, WINDOW_S)
    view = pb.Normalised(x, t, b)
    assert len(view) == len(x) and view.shape == x.shape
    np.testing.assert_array_equal(view[10:40], b.apply(t[10:40], x[10:40]))
    rows = np.array([3, 50, 120])
    np.testing.assert_array_equal(view[rows], b.apply(t[rows], x[rows]))
