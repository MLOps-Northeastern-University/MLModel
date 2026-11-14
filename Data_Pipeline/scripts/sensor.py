"""Signal processing for one wrist: the model's 10-s windows, wear, orientation, and the 7 hourly features.

Windows (config windows:, the spec Model_Pipeline and the app share):
    1. split the hour at recording gaps; nothing is filtered or resampled across one
    2. clip to the sensor range, interpolate onto an exact 100 Hz grid that starts on
       a window boundary, then resample to 30 Hz (resample_poly's low-pass removes
       everything above 15 Hz, so nothing aliases)
    3. cut whole 10-s windows: time_s is a multiple of 10 on the diary clock
    4. keep each window's per-axis mean (for the orientation check), then remove it

Orientation (config orientation:, checked on COPS's two wrists): X along the arm,
Y across the wrist, Z out of the face. A watch strapped on turned round has X
mostly positive; X and Y are flipped back. This is decided per wear bout, since
a watch can be put back on another way round. The right wrist's Y is then
flipped so every wrist reads like a left one.
"""

import math

import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt

AXES = ["X", "Y", "Z"]
FEATURE_NAMES = ["move_mean", "move_std", "move_p95", "still_fraction", "tremor_mean", "tremor_p95", "tremor_ratio"]

# ---- windows -----------------------------------------------------------------------------


def _segments(t, max_gap):
    """Index arrays of the runs of samples with no gap longer than max_gap."""
    return np.split(np.arange(len(t)), np.flatnonzero(np.diff(t) > max_gap) + 1)


def hour_windows(df, offset_s, cfg):
    """Cut one hour of one wrist into windows.

    Args:
        df: The hour's samples: Time (seconds after midnight), X, Y, Z, optional Temp.
        offset_s: Seconds from the diary clock's start to that midnight (day x 86400).

    Returns:
        dict of arrays, one entry per window: time_s (int64), mean (n, 3), signal
        (n, 3, samples) float32 with the mean removed, temp (n,).
    """
    s, w = cfg["sensor"], cfg["windows"]
    fs, rate, sec = s["sampling_rate_hz"], w["rate_hz"], w["seconds"]
    g = math.gcd(rate, fs)
    df = df.dropna(subset=["Time"] + AXES).sort_values("Time").drop_duplicates("Time")
    t = df["Time"].to_numpy(float) + offset_s
    xyz = np.clip(df[AXES].to_numpy(float), *s["accel_range_g"])
    temp = df["Temp"].to_numpy(float) if "Temp" in df else np.full(len(t), np.nan)

    starts, lows, temps = [], [], []
    for idx in _segments(t, s["max_sample_gap_seconds"]):
        if len(idx) < 2:
            continue
        # COPS stamps each sample at the end of its interval (an hour runs 10:00:00.010 to
        # 11:00:00.000), so a window may start up to one sample before the first stamp.
        t0 = math.ceil((t[idx[0]] - 1 / fs) / sec - 1e-9) * sec
        # A window needs samples up to its last grid point, one sample before its end.
        n = int((t[idx[-1]] + 1 / fs - t0 + 1e-6) // sec)
        if n <= 0:
            continue
        grid = t0 + np.arange(n * sec * fs) / fs
        hi = np.column_stack([np.interp(grid, t[idx], xyz[idx, a]) for a in range(3)])
        lo = resample_poly(hi, rate // g, fs // g, axis=0, padtype="line")[: n * sec * rate]
        lows.append(lo.reshape(n, sec * rate, 3).transpose(0, 2, 1))
        temps.append(np.interp(grid, t[idx], temp[idx]).reshape(n, -1).mean(axis=1))
        starts.append(t0 + sec * np.arange(n))

    if not starts:
        return empty_windows(cfg)
    low = np.concatenate(lows)
    mean = low.mean(axis=2)
    return {"time_s": np.concatenate(starts).astype(np.int64), "mean": mean,
            "signal": (low - mean[:, :, None]).astype(np.float32), "temp": np.concatenate(temps)}


def empty_windows(cfg):
    w = cfg["windows"]
    return {"time_s": np.zeros(0, np.int64), "mean": np.zeros((0, 3)),
            "signal": np.zeros((0, w["axes"], w["seconds"] * w["rate_hz"]), np.float32), "temp": np.zeros(0)}


def wear_flags(time_s, signal, temp, diary_worn, cfg):
    """True where the watch was worn: the diary says so, and the sensor isn't cold and still.

    The sensor test looks at a block of windows around each one, because one 10-s
    window is too short to tell a watch on a table from a resting arm.
    """
    c = cfg["wear"]
    still = (signal.std(axis=2).max(axis=1) < c["still_std_g"]).astype(float)
    block = pd.DataFrame({"still": still, "temp": temp}, index=pd.to_timedelta(time_s, unit="s"))
    block = block.rolling(f"{c['block_minutes'] * 60}s", center=True, min_periods=1).mean()
    off = (block["still"] >= c["min_still_fraction"]) & (block["temp"] < c["max_temp_c"])
    return np.asarray(diary_worn, dtype=bool) & ~off.to_numpy()

# ---- orientation -------------------------------------------------------------------------


def is_turned_round(along_arm):
    """A watch whose along-arm axis averages above 0 g was strapped on turned round."""
    along_arm = np.asarray(along_arm, dtype=float)
    return bool(len(along_arm) and np.nanmean(along_arm) > 0)


def wear_bouts(time_s, worn, gap_s):
    """A bout number for each window. A new bout starts when the watch was off for gap_s or more.

    A watch can only be strapped on another way round while it's off, so each bout
    gets its own orientation decision. Windows before the first worn one are bout 0.
    """
    last_worn = pd.Series(np.where(worn, time_s, np.nan)).ffill().to_numpy()
    previous = np.concatenate([[np.nan], last_worn[:-1]])
    return np.cumsum(worn & ~(time_s - previous < gap_s))


def turned_by_bout(means, worn, awake, bouts, cfg):
    """True for the windows of every bout whose watch was turned round.

    Each bout is judged on its worn, waking windows, or on all its worn ones when
    it has no waking ones.
    """
    along = means[:, cfg["orientation"]["along_arm_axis"]]
    turned = np.zeros(len(bouts), dtype=bool)
    for bout in np.unique(bouts[worn]):
        inside = bouts == bout
        use = inside & worn & awake
        turned[inside] = is_turned_round(along[use if use.any() else inside & worn])
    return turned


def orient(values, turned, wrist, cfg):
    """Flip axes in place (axis 1 of values) so the watch reads like a left wrist worn the usual way.

    turned is one value for all rows, or one per row (axis 0).

    Returns:
        bool: True when the mirror (orientation.mirror_wrist) was applied.
    """
    o = cfg["orientation"]
    sign = np.where(np.broadcast_to(turned, values.shape[:1]), -1.0, 1.0).reshape((-1,) + (1,) * (values.ndim - 2))
    values[:, o["along_arm_axis"]] *= sign
    values[:, o["mirror_axis"]] *= sign
    mirrored = wrist == o["mirror_wrist"]
    if mirrored:
        values[:, o["mirror_axis"]] *= -1
    return mirrored


def minute_means(df, offset_s):
    """Per-minute average of each axis (the watch's tilt), indexed by minute on the diary clock."""
    df = df.dropna(subset=["Time"] + AXES)
    return df[AXES].groupby(((df["Time"] + offset_s) // 60).astype(int).to_numpy()).mean()


def wrist_agreement(chosen, other, other_wrist, cfg):
    """How well the two wrists agree, axis by axis, once both are oriented.

    chosen: the chosen wrist's per-minute axis averages, already oriented.
    other: the other wrist's, as recorded; each day of it is oriented on its own.
    Both wrists see the same posture changes, so each axis should rise and fall
    together on both: every correlation above 0.
    """
    if chosen.empty or other.empty:
        return {"minutes": 0}
    b = other.to_numpy(copy=True)
    day = other.index.to_numpy() // (24 * 60)
    along = b[:, cfg["orientation"]["along_arm_axis"]]
    turned = np.zeros(len(b), dtype=bool)
    for d in np.unique(day):
        turned[day == d] = is_turned_round(along[day == d])
    orient(b, turned, other_wrist, cfg)
    joined = chosen.join(pd.DataFrame(b, other.index, AXES), how="inner", rsuffix="_o")
    with np.errstate(divide="ignore", invalid="ignore"):  # an axis that never moves has no correlation (NaN)
        corrs = {f"corr_{ax.lower()}": float(joined[ax].corr(joined[f"{ax}_o"])) for ax in AXES}
    return {"minutes": len(joined), **corrs}

# ---- the 7 hourly features ----------------------------------------------------------------


def bandpass(signal, low, high, fs, order=4):
    """Zero-phase Butterworth band-pass filter."""
    return sosfiltfilt(butter(order, [low, high], btype="band", fs=fs, output="sos"), signal)


def empty_features(status):
    return {**{name: np.nan for name in FEATURE_NAMES}, "valid_fraction": 0.0, "status": status}


def hour_features(df, cfg):
    """Movement and tremor features for one hour of one wrist, from the 100 Hz magnitude.

    0.1-3 Hz is everyday movement (and removes gravity), 4-8 Hz is tremor. Each band
    is filtered within a gap-free run, made positive, averaged per 1-s epoch, then
    summarised over the hour.
    """
    s, f = cfg["sensor"], cfg["features"]
    fs = s["sampling_rate_hz"]
    valid = df.dropna(subset=["Time"] + AXES).sort_values("Time")
    if len(valid) < fs * f["min_valid_seconds"]:
        return empty_features("insufficient_data")
    magnitude = np.sqrt((np.clip(valid[AXES].to_numpy(float), *s["accel_range_g"]) ** 2).sum(axis=1))
    epoch = int(fs * f["epoch_seconds"])
    move, trem = [], []
    for idx in _segments(valid["Time"].to_numpy(float), s["max_sample_gap_seconds"]):
        k = len(idx) // epoch
        if k == 0:
            continue
        for band, out in [(f["movement_band_hz"], move), (f["tremor_band_hz"], trem)]:
            filtered = np.abs(bandpass(magnitude[idx], *band, fs, f["filter_order"]))
            out.append(filtered[: k * epoch].reshape(k, epoch).mean(axis=1))
    if not move:
        return empty_features("insufficient_data")
    move, trem = np.concatenate(move), np.concatenate(trem)
    move_mean, tremor_mean = float(move.mean()), float(trem.mean())
    return {
        "move_mean": move_mean, "move_std": float(move.std()), "move_p95": float(np.percentile(move, 95)),
        "still_fraction": float((move < f["still_threshold_g"]).mean()),
        "tremor_mean": tremor_mean, "tremor_p95": float(np.percentile(trem, 95)),
        "tremor_ratio": tremor_mean / move_mean if move_mean > 0 else np.nan,
        "valid_fraction": min(1.0, len(valid) / (fs * f["expected_seconds_per_file"])),
        "status": "ok",
    }
