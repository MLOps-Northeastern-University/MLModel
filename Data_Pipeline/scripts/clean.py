"""Filter hours down to valid, labelled, worn training examples."""

import pandas as pd

from scripts.utils.logging_setup import get_logger

logger = get_logger("clean")


def clean_hours(df, cfg):
    """Apply the cleaning rules and record why each hour was removed.

    Rules (in order):
        1. no kinesia label (Sleep, No Data)
        2. diary says the watch was not worn
        3. missing or unusable sensor data on a required wrist
        4. too little valid data in the hour (covers partially worn hours)
        5. sensor-based non-wear: cold watch that never moves

    Returns:
        (DataFrame, dict): The cleaned rows and the count removed per reason.
    """
    c = cfg["cleaning"]
    wrists = ["L", "R"] if c["require_both_wrists"] else []
    reasons = pd.Series("", index=df.index, dtype=object)

    def mark(mask, reason):
        reasons[mask & (reasons == "")] = reason

    mark(df["kinesia_class"].isna(), "no_label")
    mark(df["wear_status"].eq("Wearable Not Worn"), "not_worn")
    for w in wrists:
        status = df.get(f"{w}_status", pd.Series(index=df.index, dtype=object))
        mark(status.ne("ok"), f"{w}_no_sensor_data")
        frac = pd.to_numeric(df.get(f"{w}_valid_fraction"), errors="coerce")
        mark(frac < c["min_valid_fraction"], f"{w}_low_valid_fraction")
        temp = pd.to_numeric(df.get(f"{w}_temp_mean"), errors="coerce")
        still = pd.to_numeric(df.get(f"{w}_still_fraction"), errors="coerce")
        mark((temp < c["nonwear_max_temp_c"]) & (still > c["nonwear_min_still_fraction"]),
             f"{w}_sensor_nonwear")

    removed = reasons[reasons != ""].value_counts().to_dict()
    clean = df[reasons == ""].copy()
    logger.info("Cleaning kept %d of %d hours; removed %s", len(clean), len(df), removed)
    return clean.reset_index(drop=True), {"kept": len(clean), "total": len(df), "removed": removed}


def drop_duplicate_hours(df):
    """Keep one row per participant-hour."""
    before = len(df)
    df = df.drop_duplicates(subset=["participant_id", "hour_index"], keep="first")
    if len(df) < before:
        logger.warning("Dropped %d duplicate participant-hours", before - len(df))
    return df
