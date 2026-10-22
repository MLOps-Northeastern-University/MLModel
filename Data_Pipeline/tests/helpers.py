"""Build small synthetic COPS-style files for tests.

Nothing here comes from the real dataset. Each fake "hour" is only
2 minutes long so tests run in a fraction of a second.
"""

import io
import zipfile
from functools import partial

import numpy as np
import pandas as pd

from scripts.utils.config import load_config

FS = 100
SECONDS = 120
_COLUMNS_BEFORE_DOSES = [
    "ID", "Day", "Time", "Visit", "Kinesia", "KinesiaScore", "Tremor", "TremorScore",
    "Freezing", "FreezingScore", "Fall", "FallScore", "Medication",
]
_COLUMNS_AFTER_DOSES = [
    "WearableDataAvailability", "WearableDataLeftCSV", "WearableDataLeftZIP",
    "WearableDataRightCSV", "WearableDataRightZIP",
]


def dose_column_names(dose_columns=1):
    """Real diaries have either one "Dosage" column or "Dosage_1".."Dosage_4"."""
    if dose_columns == 1:
        return ["Dosage"]
    return [f"Dosage_{i}" for i in range(1, dose_columns + 1)]


def diary_columns(dose_columns=1):
    return _COLUMNS_BEFORE_DOSES + dose_column_names(dose_columns) + _COLUMNS_AFTER_DOSES


DIARY_COLUMNS = diary_columns()


def make_config(tmp_path, **sections):
    """Config pointing all outputs at tmp_path, with short test hours."""
    overrides = {
        "project_root": str(tmp_path),
        "features": {"expected_seconds_per_file": SECONDS, "min_valid_seconds": 30},
        "anomalies": {"min_hours_per_participant": 1},
        "bias": {"min_participants_for_model": 2, "min_slice_rows": 1},
    }
    for key, value in sections.items():
        overrides.setdefault(key, {}).update(value)
    return load_config(overrides=overrides)


def signal(kind="still", seconds=SECONDS, seed=0):
    """Fake wrist signal: 'still', 'moving' (1 Hz) or 'tremor' (5 Hz)."""
    rng = np.random.default_rng(seed)
    t = np.arange(seconds * FS) / FS
    x = np.full_like(t, 0.45)
    y = np.full_like(t, 0.74)
    z = np.full_like(t, -0.49)
    if kind == "moving":
        x = x + 0.3 * np.sin(2 * np.pi * 1.0 * t)
    if kind == "tremor":
        x = x + 0.15 * np.sin(2 * np.pi * 5.0 * t)
    x = x + rng.normal(0, 0.002, len(t))
    clock = [f"11:{int(s // 60):02d}:{s % 60:06.3f}" for s in t + 0.01]
    return pd.DataFrame({"Time": clock, "X": x.round(4), "Y": y.round(4), "Z": z.round(4),
                         "Photo": 60, "Temp": 31.0})


def csv_bytes(df):
    return df.to_csv(sep=";", index=False).encode()


def inner_zip(df, csv_name):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(csv_name, csv_bytes(df))
    return buf.getvalue()


def diary_row(pid, day, hour_end, kinesia, score, wear="Wearable Worn", tremor="No Tremor",
              tremor_score=0, medication=None, doses=None, files=True, visit="outpatient",
              dose_columns=1):
    stem = f"{pid}_Day{day}_{hour_end - 1}h-{hour_end}h"
    doses = doses if doses is not None else ([100] if medication else [])
    row = {
        "ID": pid, "Day": f"{day} days", "Time": f"{hour_end} hr", "Visit": visit,
        "Kinesia": kinesia, "KinesiaScore": score, "Tremor": tremor, "TremorScore": tremor_score,
        "Freezing": "No Freezing", "FreezingScore": 0, "Fall": "No Fall", "FallScore": 0,
        "Medication": medication,
        "WearableDataAvailability": wear,
        "WearableDataLeftCSV": f"{stem}_leftWrist.csv" if files else None,
        "WearableDataLeftZIP": f"{stem}_leftWrist.zip" if files else None,
        "WearableDataRightCSV": f"{stem}_rightWrist.csv" if files else None,
        "WearableDataRightZIP": f"{stem}_rightWrist.zip" if files else None,
    }
    for i, name in enumerate(dose_column_names(dose_columns)):
        row[name] = doses[i] if i < len(doses) else None
    return row


def default_hours(pid, dose_columns=1):
    """(diary row, signal kind) pairs covering the main label types.

    With numbered dose columns, the first hour has two medications in one
    quoted field, like the real COPS-11 diary.
    """
    if dose_columns == 1:
        meds = {"medication": "Levodopa / Carbidopa", "doses": [100]}
    else:
        meds = {"medication": "Levodopa / Benserazid;Opicapone", "doses": [50, 50]}
    row = partial(diary_row, dose_columns=dose_columns)
    return [
        (row(pid, 0, 9, "Good Kinesia", 0, **meds), "moving"),
        (row(pid, 0, 10, "Slight Akinesia", -1), "still"),
        (row(pid, 0, 11, "Slight Dyskinesia", 1, tremor="Weak Tremor", tremor_score=1), "tremor"),
        (row(pid, 0, 12, "Discomforting Dyskinesia", 2), "moving"),
        (row(pid, 0, 13, "Sleep", None), "still"),
        (row(pid, 0, 14, "No Data", 1), "still"),  # text and score disagree
        (row(pid, 0, 15, "Slight Dyskinesia", 1, wear="Wearable Not Worn", files=False), None),
    ]


def make_participant_zip(folder, pid="COPS-99", hours=None, include_updrs=True, corrupt=None,
                         dose_columns=1):
    """Write a participant zip shaped like the real ones and return its path."""
    hours = hours or default_hours(pid, dose_columns)
    path = folder / f"{pid}.zip"
    with zipfile.ZipFile(path, "w") as z:
        diary = pd.DataFrame([row for row, _ in hours], columns=diary_columns(dose_columns))
        z.writestr(f"{pid}/{pid}_symptomdiary.csv", csv_bytes(diary))
        if include_updrs:
            for state, total in [("OFF", 41), ("ON", 20)]:
                updrs = pd.DataFrame([{"DaysSinceExamination": 139, "MedicationState": int(state == "ON"),
                                       "MedicationDose": None, "DBSState": 0, "TotalScore": total}])
                z.writestr(f"{pid}/{pid}_UPDRS_{state}.csv", csv_bytes(updrs))
        for i, (row, kind) in enumerate(hours):
            if kind is None:
                continue
            for side in ["Left", "Right"]:
                zip_name = row[f"WearableData{side}ZIP"]
                if corrupt == zip_name:
                    z.writestr(f"{pid}/Accelerometry/{zip_name}", b"not a zip")
                    continue
                df = signal(kind, seed=i)
                z.writestr(f"{pid}/Accelerometry/{zip_name}", inner_zip(df, row[f"WearableData{side}CSV"]))
    return path


def make_demographics(folder, pids):
    rows = [{"ID": pid, "Age": 50 + 7 * i, "Sex": "M" if i % 2 else "F", "DBS": "No"}
            for i, pid in enumerate(pids)]
    path = folder / "Demographics.csv"
    pd.DataFrame(rows).to_csv(path, sep=";", index=False)
    return path
