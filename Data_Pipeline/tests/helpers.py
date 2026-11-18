"""Small synthetic COPS-style files for tests.

Nothing here comes from the real dataset. Each fake "hour" is 2 minutes long,
recorded at the start of its clock hour, so tests run in a fraction of a second.
"""

import io
import zipfile
from functools import partial

import numpy as np
import pandas as pd

from scripts.utils import PROJECT_ROOT, load_config

FS = 100
SECONDS = 120
_BEFORE_DOSES = ["ID", "Day", "Time", "Visit", "Kinesia", "KinesiaScore", "Tremor", "TremorScore",
                 "Freezing", "FreezingScore", "Fall", "FallScore", "Medication"]
_AFTER_DOSES = ["WearableDataAvailability", "WearableDataLeftCSV", "WearableDataLeftZIP",
                "WearableDataRightCSV", "WearableDataRightZIP"]


def dose_column_names(dose_columns=1):
    """Real diaries have either one "Dosage" column or "Dosage_1".."Dosage_4"."""
    return ["Dosage"] if dose_columns == 1 else [f"Dosage_{i}" for i in range(1, dose_columns + 1)]


def diary_columns(dose_columns=1):
    return _BEFORE_DOSES + dose_column_names(dose_columns) + _AFTER_DOSES


def make_config(tmp_path, **sections):
    """Config with every output under tmp_path, for 2-minute test hours."""
    overrides = {
        "project_root": str(tmp_path),
        "data": {"mode": "list", "participants": [], "exclude": []},
        "run": {"parallel_participants": 1},
        "features": {"expected_seconds_per_file": SECONDS, "min_valid_seconds": 30},
        "checks": {"min_labelled_hours": 1},
        "medication": {"ledd_table": str(PROJECT_ROOT / "config" / "ledd.yaml")},
        "download": {"backoff_seconds": 0},
    }
    for key, value in sections.items():
        overrides.setdefault(key, {}).update(value)
    return load_config(overrides=overrides)


def signal(kind="still", hour=10, seconds=SECONDS, seed=0, side="left", temp=31.0):
    """A fake wrist: 'still', 'moving' (1 Hz) or 'tremor' (5 Hz), from hour:00:00.

    The watch is worn the usual way: X (along the arm) averages below 0. The right
    wrist is the left one's mirror image, so its Y is flipped.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(seconds * FS) / FS
    x, y, z = np.full_like(t, -0.45), np.full_like(t, 0.74), np.full_like(t, -0.49)
    if kind == "moving":
        x = x + 0.3 * np.sin(2 * np.pi * 1.0 * t)
    if kind == "tremor":
        x = x + 0.15 * np.sin(2 * np.pi * 5.0 * t)
    x = x + rng.normal(0, 0.002, len(t))
    y = -y if side == "right" else y
    clock = [f"{hour:02d}:{int(s // 60):02d}:{s % 60:06.3f}" for s in t + 0.01]
    return pd.DataFrame({"Time": clock, "X": x.round(4), "Y": y.round(4), "Z": z.round(4),
                         "Photo": 60, "Temp": temp})


def csv_bytes(df):
    return df.to_csv(sep=";", index=False).encode()


def inner_zip(df, csv_name):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(csv_name, csv_bytes(df))
    return buf.getvalue()


def diary_row(pid, day, hour_end, kinesia, score, wear="Wearable Worn", tremor="No Tremor",
              tremor_score=0, medication=None, doses=None, files=True, visit="outpatient", dose_columns=1):
    stem = f"{pid}_Day{day}_{hour_end - 1}h-{hour_end}h"
    doses = doses if doses is not None else ([100] if medication else [])
    row = {
        "ID": pid, "Day": f"{day} days", "Time": f"{hour_end} hr", "Visit": visit,
        "Kinesia": kinesia, "KinesiaScore": score, "Tremor": tremor, "TremorScore": tremor_score,
        "Freezing": "No Freezing", "FreezingScore": 0, "Fall": "No Fall", "FallScore": 0,
        "Medication": medication, "WearableDataAvailability": wear,
        "WearableDataLeftCSV": f"{stem}_leftWrist.csv" if files else None,
        "WearableDataLeftZIP": f"{stem}_leftWrist.zip" if files else None,
        "WearableDataRightCSV": f"{stem}_rightWrist.csv" if files else None,
        "WearableDataRightZIP": f"{stem}_rightWrist.zip" if files else None,
    }
    for i, name in enumerate(dose_column_names(dose_columns)):
        row[name] = doses[i] if i < len(doses) else None
    return row


def default_hours(pid, dose_columns=1):
    """(diary row, signal kind) pairs covering the main kinds of hour."""
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
        (row(pid, 0, 14, "No Data", 1), "still"),  # a score without a label
        (row(pid, 0, 15, "Slight Dyskinesia", 1, wear="Wearable Not Worn", files=False), None),
    ]


def make_participant_zip(folder, pid="COPS-99", hours=None, corrupt=None, dose_columns=1, temp=31.0,
                         sides=("Left", "Right")):
    """Write a participant zip shaped like the real ones and return its path."""
    hours = hours or default_hours(pid, dose_columns)
    path = folder / f"{pid}.zip"
    with zipfile.ZipFile(path, "w") as z:
        diary = pd.DataFrame([row for row, _ in hours], columns=diary_columns(dose_columns))
        z.writestr(f"{pid}/{pid}_symptomdiary.csv", csv_bytes(diary))
        for i, (row, kind) in enumerate(hours):
            if kind is None:
                continue
            hour = int(row["Time"].split()[0]) - 1
            for side in sides:
                zip_name = row[f"WearableData{side}ZIP"]
                if corrupt == zip_name:
                    z.writestr(f"{pid}/Accelerometry/{zip_name}", b"not a zip")
                    continue
                df = signal(kind, hour=hour, seed=i, side=side.lower(), temp=temp)
                z.writestr(f"{pid}/Accelerometry/{zip_name}", inner_zip(df, row[f"WearableData{side}CSV"]))
    return path


def make_local_run(tmp_path, pids=("COPS-1", "COPS-2"), sides=("left", "right"), **sections):
    """Synthetic zips plus Demographics.csv in data/raw/, and a config listing those participants."""
    folder = raw_dir(tmp_path)
    for pid in pids:
        make_participant_zip(folder, pid)
    make_demographics(folder, list(pids), list(sides))
    data = {"mode": "list", "participants": list(pids), **sections.pop("data", {})}
    return make_config(tmp_path, data=data, **sections)


def raw_dir(tmp_path):
    """The test run's data/raw/, where local zips go."""
    folder = tmp_path / "data" / "raw"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def make_demographics(folder, pids, sides=None):
    """Demographics.csv with the real file's headers. sides: each participant's more affected side."""
    sides = sides or ["left"] * len(pids)
    rows = [{"ID": pid, "Age": 50 + 7 * i, "Sex": "male" if i % 2 else "female", "Handedness": "right",
             "PD_Subtype": "akinetic-rigid", "PD_DominantSide": side, "PD_HoehnAndYahr": 2 + i % 2,
             "PD_YearsSinceDiagnosis": 8, "DBS": "yes" if i % 2 else "no"}
            for i, (pid, side) in enumerate(zip(pids, sides))]
    path = folder / "Demographics.csv"
    pd.DataFrame(rows).to_csv(path, sep=";", index=False)
    return path
