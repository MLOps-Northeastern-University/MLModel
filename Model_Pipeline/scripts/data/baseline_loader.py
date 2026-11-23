"""The hourly baseline's features, from the hours table (W_ columns) and the dose events.

One row per diary hour, with:
    z_<feature>        the 7 W_ features (logged where skewed) relative to the patient's own
                       PAST good-sensor hours (this hour left out), shrunk towards the cohort
                       while the patient has few: (x - mean_h) / sd_h, with
                           mean_h = (sum of past x + k * cohort_mean) / (n + k)
                           var_h  = (sum of past x^2 + k * (cohort_sd^2 + cohort_mean^2)) / (n + k) - mean_h^2
                       k = warmup_hours. 0 (the patient's normal) where this hour's sensor isn't good.
    sensor_ok          1 when this hour's sensor data is good enough to use
    hours_since_dose   from the end of the hour, capped; no_dose_yet flags none so far
    last_dose_ledd     LEDD of the last dose time (doses at the same time summed)
    ledd_recent        LEDD taken in the last recent_ledd_hours
    tod_sin, tod_cos   the hour of day on a circle

Only what is known by the end of each hour is used, and no labels, so the phone can compute
the same values live. The cohort mean and SD come from the training patients and are saved
with the model (models/hourly_logistic.py).
"""

import numpy as np
import pandas as pd

HOUR = 3600
FEATURE_NAMES = ["move_mean", "move_std", "move_p95", "still_fraction", "tremor_mean", "tremor_p95", "tremor_ratio"]
Z_COLUMNS = [f"z_{name}" for name in FEATURE_NAMES]
DOSE_COLUMNS = ["hours_since_dose", "no_dose_yet", "last_dose_ledd", "ledd_recent"]
TIME_COLUMNS = ["tod_sin", "tod_cos"]
FEATURE_COLUMNS = Z_COLUMNS + ["sensor_ok"] + DOSE_COLUMNS + TIME_COLUMNS
KEEP = ["participant_id", "day", "hour_start", "hour_index", "kinesia_text", "kinesia_score", "tremor_score",
        "drop_reason"]
DEFAULTS = {
    "log_features": ["move_mean", "move_std", "move_p95", "tremor_mean", "tremor_p95", "tremor_ratio"],
    "log_epsilon": 1.0e-4,
    "warmup_hours": 12,          # k: the cohort counts as this many hours of the patient's own
    "min_valid_fraction": 0.8,   # a good-sensor hour: W_status ok and at least these fractions
    "min_worn_fraction": 0.5,
    "max_hours_since_dose": 12,
    "recent_ledd_hours": 4,
}
MIN_SD = 1e-6


def settings(overrides=None):
    unknown = set(overrides or {}) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown baseline feature setting(s) {sorted(unknown)}; known: {sorted(DEFAULTS)}")
    return {**DEFAULTS, **(overrides or {})}


def good_sensor(hours, b):
    """Hours whose sensor data is good enough to use (label-free)."""
    return (hours["W_status"].eq("ok") & (hours["W_valid_fraction"] >= b["min_valid_fraction"])
            & (hours["worn_fraction"].fillna(0) >= b["min_worn_fraction"])).to_numpy()


def transformed(hours, b):
    """The 7 W_ features, logged where skewed, NaN where the sensor isn't good: (n, 7)."""
    x = np.column_stack([hours[f"W_{n}"].astype(float).to_numpy() for n in FEATURE_NAMES])
    for j, name in enumerate(FEATURE_NAMES):
        if name in b["log_features"]:
            x[:, j] = np.log(np.clip(x[:, j], 0, None) + b["log_epsilon"])
    x[~good_sensor(hours, b)] = np.nan
    return x


def cohort_stats(hours, b):
    """(mean, sd) of each transformed feature over every good hour: the training patients' normal."""
    x = transformed(hours, b)
    return np.nanmean(x, axis=0), np.maximum(np.nanstd(x, axis=0), MIN_SD)


def patient_z_scores(hours, b, cohort):
    """z of each hour against the patient's past good hours, shrunk towards the cohort. Rows in hours order."""
    x = transformed(hours, b)
    c_mean, c_sd = cohort
    k = float(b["warmup_hours"])
    z = np.zeros_like(x)
    order = np.lexsort((hours["hour_index"].to_numpy(), hours["participant_id"].astype(str).to_numpy()))
    pid = hours["participant_id"].astype(str).to_numpy()[order]
    starts = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
    for start, stop in zip(starts, np.r_[starts[1:], len(order)]):
        rows = order[start:stop]
        v = x[rows]
        has = ~np.isnan(v)
        filled = np.where(has, v, 0.0)
        # Sums over the hours BEFORE each one: cumulative sums shifted down by one row.
        n = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(has, axis=0)[:-1]])
        s = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(filled, axis=0)[:-1]])
        s2 = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(filled ** 2, axis=0)[:-1]])
        mean = (s + k * c_mean) / (n + k)
        var = (s2 + k * (c_sd ** 2 + c_mean ** 2)) / (n + k) - mean ** 2
        sd = np.sqrt(np.maximum(var, MIN_SD ** 2))
        z[rows] = np.where(has, (filled - mean) / sd, 0.0)
    return pd.DataFrame(z, columns=Z_COLUMNS, index=hours.index)


def dose_features(hours, events, b):
    """Hours since the last dose, its LEDD, and recent LEDD, all at the end of each hour."""
    events = events.reindex(columns=["participant_id", "time_s", "kind", "ledd"])   # no doses at all is fine
    doses = (events[events["kind"] == "dose"].astype({"participant_id": str, "time_s": "int64", "ledd": float})
             .groupby(["participant_id", "time_s"], as_index=False)["ledd"].sum().sort_values("time_s"))
    end = ((hours["hour_index"] + 1) * HOUR).astype("int64")
    left = pd.DataFrame({"participant_id": hours["participant_id"].astype(str), "end": end, "row": hours.index})
    last = pd.merge_asof(left.sort_values("end"), doses.rename(columns={"time_s": "dose_s", "ledd": "last_dose_ledd"}),
                         left_on="end", right_on="dose_s", by="participant_id", direction="backward").set_index("row")
    last = last.reindex(hours.index)
    since = (last["end"] - last["dose_s"]) / HOUR
    out = pd.DataFrame(index=hours.index)
    out["no_dose_yet"] = since.isna().astype(int)
    out["hours_since_dose"] = since.clip(upper=b["max_hours_since_dose"]).fillna(b["max_hours_since_dose"])
    out["last_dose_ledd"] = last["last_dose_ledd"].fillna(0.0)
    window = b["recent_ledd_hours"] * HOUR
    recent = np.zeros(len(hours))
    for pid, rows in hours.groupby(hours["participant_id"].astype(str)).groups.items():
        p = doses[doses["participant_id"] == pid]
        t, ledd = p["time_s"].to_numpy(), np.concatenate([[0.0], p["ledd"].to_numpy().cumsum()])
        e = end.loc[rows].to_numpy()
        taken = ledd[np.searchsorted(t, e, "right")] - ledd[np.searchsorted(t, e - window, "right")]
        recent[hours.index.get_indexer(rows)] = taken
    out["ledd_recent"] = recent
    return out


def time_of_day(hours):
    angle = 2 * np.pi * (hours["hour_start"] + 0.5) / 24  # the hour's middle
    return pd.DataFrame({"tod_sin": np.sin(angle), "tod_cos": np.cos(angle)}, index=hours.index)


def feature_matrix(hours, events, b, cohort):
    """FEATURE_COLUMNS for every hour, in hours' order and index."""
    sensor_ok = pd.Series(good_sensor(hours, b).astype(int), index=hours.index, name="sensor_ok")
    return pd.concat([patient_z_scores(hours, b, cohort), sensor_ok, dose_features(hours, events, b),
                      time_of_day(hours)], axis=1)[FEATURE_COLUMNS]


def build_features(hours, events, b=None, cohort=None):
    """KEEP columns (ids, labels, drop_reason) plus FEATURE_COLUMNS, sorted by patient and hour.

    cohort defaults to these hours' own stats: fine for looking at the data, but a model must pass
    its training patients' cohort.
    """
    b = settings(b)
    hours = hours.reset_index(drop=True)
    cohort = cohort if cohort is not None else cohort_stats(hours, b)
    table = pd.concat([hours[[c for c in KEEP if c in hours]], feature_matrix(hours, events, b, cohort)], axis=1)
    return table.sort_values(["participant_id", "hour_index"]).reset_index(drop=True)


def load_baseline(cfg, b=None):
    """Read hours.csv and events.csv (config paths) and build the table, for looking at the data."""
    paths = cfg["resolved_paths"]
    return build_features(pd.read_csv(paths["hours_csv"], low_memory=False), pd.read_csv(paths["events_csv"]), b)
