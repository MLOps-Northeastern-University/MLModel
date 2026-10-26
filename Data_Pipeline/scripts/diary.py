"""Parse a participant's hourly symptom diary into clean labels.

The raw diary has one row per hour. Each label has a text column (e.g.
"Slight Dyskinesia") and a score column (e.g. 1). In the real data these
sometimes disagree, so the text is treated as the source of truth and every
disagreement is recorded as a label issue.

Time convention: a row with Time "18 hr" describes the hour 17:00-18:00,
matching the accelerometry file "..._17h-18h_...".

Dose columns differ between participants: some diaries have one "Dosage"
column, others have "Dosage_1" to "Dosage_4", one per medication taken in
the same hour. Both layouts are accepted.
"""

import re

import numpy as np
import pandas as pd

from scripts.utils.file_io import read_text_csv
from scripts.utils.logging_setup import get_logger

logger = get_logger("diary")

REQUIRED_COLUMNS = [
    "ID", "Day", "Time", "Visit", "Kinesia", "KinesiaScore", "Tremor", "TremorScore",
    "Freezing", "FreezingScore", "Fall", "FallScore", "Medication",
    "WearableDataAvailability", "WearableDataLeftCSV", "WearableDataLeftZIP",
    "WearableDataRightCSV", "WearableDataRightZIP",
]
DOSAGE_COLUMN = re.compile(r"^Dosage(?:_(\d+))?$")
BINARY_TEXT = {
    "No Freezing": 0, "Freezing": 1,
    "No Fall": 0, "Fall": 1,
}


class DiaryError(ValueError):
    """Raised when a diary file is missing required columns."""


def load_diary(source, sep=";"):
    """Read a raw diary CSV (path or file object), UTF-8 or Windows-encoded."""
    return read_text_csv(source, sep=sep, na_values=["NaN", ""], keep_default_na=True)


def _first_int(value):
    match = re.search(r"-?\d+", str(value))
    return int(match.group()) if match else np.nan


def reconcile(text, score, mapping):
    """Return a trusted score for one label and an issue code, if any.

    Rules:
        - text maps to a score and the score matches -> keep it
        - text maps to a score but the score differs or is missing -> use the text
        - text has no score (Sleep, No Data) but a score is present -> drop it
    """
    text = None if pd.isna(text) else str(text).strip()
    has_score = not pd.isna(score)
    if text in mapping:
        expected = mapping[text]
        if not has_score:
            return expected, "score_missing"
        if float(score) != float(expected):
            return expected, "score_text_mismatch"
        return expected, None
    if has_score:
        return np.nan, "score_without_label"
    return np.nan, None


def dosage_columns(columns):
    """Return the dose columns in order: ["Dosage"] or ["Dosage_1", "Dosage_2", ...]."""
    found = []
    for col in columns:
        match = DOSAGE_COLUMN.match(str(col))
        if match:
            found.append((int(match.group(1) or 0), col))
    return [col for _, col in sorted(found)]


def join_doses(row, columns):
    """Join one hour's doses as text, e.g. "50" or "50;50" (NaN if none).

    The order matches the ";"-separated Medication field. Doses are kept as
    text so every participant gives the same column type.
    """
    doses = []
    for col in columns:
        value = row[col]
        if pd.isna(value):
            continue
        number = pd.to_numeric(value, errors="coerce")
        doses.append(str(value).strip() if pd.isna(number) else f"{number:g}")
    return ";".join(doses) if doses else np.nan


def map_kinesia_class(score, classes):
    """Map a -3..+3 kinesia score to 'off', 'on' or 'dyskinetic'."""
    if pd.isna(score):
        return None
    for label, values in classes.items():
        if int(score) in values:
            return label
    return None


def parse_diary(raw, cfg):
    """Turn a raw diary into one clean row per hour.

    Args:
        raw: DataFrame from load_diary.
        cfg: Pipeline configuration.

    Returns:
        (DataFrame, list[dict]): The parsed diary and a list of label issues.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise DiaryError(f"Diary is missing columns: {missing}")
    if raw.empty:
        raise DiaryError("Diary has no rows")
    dose_cols = dosage_columns(raw.columns)
    if not dose_cols:
        logger.warning("Diary has no Dosage column; dosage_mg will be empty")

    labels = cfg["labels"]
    rows, issues = [], []
    for idx, r in raw.iterrows():
        day = _first_int(r["Day"])
        hour_end = _first_int(r["Time"])
        hour_start = hour_end - 1 if not pd.isna(hour_end) else np.nan

        kin_score, kin_issue = reconcile(r["Kinesia"], r["KinesiaScore"], labels["kinesia_text_to_score"])
        trem_score, trem_issue = reconcile(r["Tremor"], r["TremorScore"], labels["tremor_text_to_score"])
        frz_score, frz_issue = reconcile(r["Freezing"], r["FreezingScore"], BINARY_TEXT)
        fall_score, fall_issue = reconcile(r["Fall"], r["FallScore"], BINARY_TEXT)

        for field, issue in [("kinesia", kin_issue), ("tremor", trem_issue),
                             ("freezing", frz_issue), ("fall", fall_issue)]:
            if issue:
                issues.append({"participant_id": r["ID"], "day": day, "hour_start": hour_start,
                               "field": field, "issue": issue, "row": int(idx)})

        rows.append({
            "participant_id": str(r["ID"]).strip(),
            "day": day,
            "hour_start": hour_start,
            "hour_index": day * 24 + hour_start if not pd.isna(day) else np.nan,
            "visit": r["Visit"],
            "kinesia_text": r["Kinesia"],
            "kinesia_score": kin_score,
            "kinesia_class": map_kinesia_class(kin_score, labels["kinesia_classes"]),
            "tremor_score": trem_score,
            "tremor_present": None if pd.isna(trem_score) else int(trem_score > 0),
            "freezing": frz_score,
            "fall": fall_score,
            "medication_taken": int(not pd.isna(r["Medication"])),
            "medication": r["Medication"],
            "dosage_mg": join_doses(r, dose_cols),
            "wear_status": r["WearableDataAvailability"],
            "left_zip": r["WearableDataLeftZIP"],
            "left_csv": r["WearableDataLeftCSV"],
            "right_zip": r["WearableDataRightZIP"],
            "right_csv": r["WearableDataRightCSV"],
        })

    diary = pd.DataFrame(rows)
    diary = add_hours_since_dose(diary)
    logger.info(
        "Parsed diary for %s: %d hours, %d labelled, %d label issues",
        diary["participant_id"].iloc[0] if len(diary) else "?",
        len(diary), diary["kinesia_class"].notna().sum(), len(issues),
    )
    return diary, issues


def add_hours_since_dose(diary):
    """Add hours since the most recent medication dose (NaN before the first)."""
    diary = diary.sort_values("hour_index").reset_index(drop=True)
    last_dose, values = np.nan, []
    for _, r in diary.iterrows():
        if r["medication_taken"]:
            last_dose = r["hour_index"]
        values.append(r["hour_index"] - last_dose if not np.isnan(last_dose) else np.nan)
    diary["hours_since_dose"] = values
    return diary
