"""A participant's hourly symptom diary as one clean row per hour. No hour is dropped.

Each label has a text column ("Slight Akinesia") and a score column (-1). They
sometimes disagree; the text is trusted and every disagreement is recorded.
Sleep and No Data hours have no score: Model_Pipeline labels Sleep hours asleep
from kinesia_text and leaves No Data hours unlabelled.

Clock: Day "0 days" is the first diary day. A row with Time "18 hr" is the hour
17:00-18:00, so hour_start is 17 and hour_index = day x 24 + hour_start, the
hours since the first diary day began.

Doses: some diaries have one "Dosage" column, others "Dosage_1".."Dosage_4", one
per medication taken that hour; they're joined in Medication's order ("50;50").
"""

import re

import numpy as np
import pandas as pd

from data_pipeline.utils import get_logger

logger = get_logger("diary")

NOT_WORN = "Wearable Not Worn"
SLEEP = "Sleep"
REQUIRED_COLUMNS = ["ID", "Day", "Time", "Visit", "Kinesia", "KinesiaScore", "Tremor", "TremorScore",
                    "Medication", "WearableDataAvailability", "WearableDataLeftCSV", "WearableDataLeftZIP",
                    "WearableDataRightCSV", "WearableDataRightZIP"]
DOSAGE_COLUMN = re.compile(r"^Dosage(?:_(\d+))?$")


class DiaryError(ValueError):
    """The diary is missing required columns or has no rows."""


def _first_int(value):
    match = re.search(r"-?\d+", str(value))
    return int(match.group()) if match else np.nan


def _text(value):
    return None if pd.isna(value) else str(value).strip()


def reconcile(text, score, mapping):
    """A trusted score for one label, and an issue code if the score had to be corrected."""
    has_score = not pd.isna(score)
    if text in mapping:
        expected = mapping[text]
        if not has_score:
            return expected, "score_missing"
        return expected, None if float(score) == float(expected) else "score_text_mismatch"
    return np.nan, "score_without_label" if has_score else None


def dosage_columns(columns):
    """The dose columns in order: ["Dosage"] or ["Dosage_1", "Dosage_2", ...]."""
    found = [(int(m.group(1) or 0), c) for c in columns if (m := DOSAGE_COLUMN.match(str(c)))]
    return [c for _, c in sorted(found)]


def join_doses(row, columns):
    """One hour's doses as text, e.g. "50" or "50;50" (NaN if none)."""
    doses = []
    for col in columns:
        value = row[col]
        if pd.isna(value):
            continue
        number = pd.to_numeric(value, errors="coerce")
        doses.append(str(value).strip() if pd.isna(number) else f"{number:g}")
    return ";".join(doses) if doses else np.nan


def parse_diary(raw, cfg):
    """Turn a raw diary into one row per hour.

    Returns:
        (DataFrame, list[dict]): the hours, sorted by hour_index, and the label issues.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise DiaryError(f"Diary is missing columns: {missing}")
    if raw.empty:
        raise DiaryError("Diary has no rows")
    dose_cols = dosage_columns(raw.columns)
    labels = cfg["labels"]

    rows, issues = [], []
    for idx, r in raw.iterrows():
        pid, day, hour_start = str(r["ID"]).strip(), _first_int(r["Day"]), _first_int(r["Time"]) - 1
        kinesia_text, tremor_text = _text(r["Kinesia"]), _text(r["Tremor"])
        kinesia, kinesia_issue = reconcile(kinesia_text, r["KinesiaScore"], labels["kinesia_text_to_score"])
        tremor, tremor_issue = reconcile(tremor_text, r["TremorScore"], labels["tremor_text_to_score"])
        for field, issue in [("kinesia", kinesia_issue), ("tremor", tremor_issue)]:
            if issue:
                issues.append({"participant_id": pid, "day": day, "hour_start": hour_start,
                               "field": field, "issue": issue, "row": int(idx)})
        rows.append({
            "participant_id": pid, "day": day, "hour_start": hour_start, "hour_index": day * 24 + hour_start,
            "visit": _text(r["Visit"]), "kinesia_text": kinesia_text, "kinesia_score": kinesia,
            "tremor_score": tremor, "wear_status": _text(r["WearableDataAvailability"]),
            "medication": _text(r["Medication"]), "dosage_mg": join_doses(r, dose_cols),
            "left_zip": r["WearableDataLeftZIP"], "left_csv": r["WearableDataLeftCSV"],
            "right_zip": r["WearableDataRightZIP"], "right_csv": r["WearableDataRightCSV"],
        })

    hours = pd.DataFrame(rows)
    if hours["hour_index"].isna().any():
        raise DiaryError("Diary has a Day or Time that isn't a number")
    hours = hours.astype({"day": int, "hour_start": int, "hour_index": int})
    dupes = hours.duplicated(["participant_id", "hour_index"])
    if dupes.any():
        logger.warning("%s: dropped %d duplicate diary hours", hours["participant_id"].iloc[0], dupes.sum())
    hours = hours[~dupes].sort_values("hour_index").reset_index(drop=True)
    logger.info("Diary %s: %d hours, %d with a kinesia score, %d sleep, %d label issues",
                hours["participant_id"].iloc[0], len(hours), hours["kinesia_score"].notna().sum(),
                (hours["kinesia_text"] == SLEEP).sum(), len(issues))
    return hours, issues
