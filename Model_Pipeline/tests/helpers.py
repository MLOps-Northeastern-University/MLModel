"""Build a small synthetic training table for tests.

Nothing here comes from the real dataset. The table has the columns of the
data pipeline's data/processed/hourly_features.csv (one wrist, W_* features),
with feature values in the real data's range, so code that reads the real
file can be tested without it.
"""

import numpy as np
import pandas as pd

from scripts.utils.config import load_config
from scripts.utils.file_io import write_table

FEATURE_NAMES = ["move_mean", "move_std", "move_p95", "still_fraction", "tremor_mean", "tremor_p95", "tremor_ratio"]
QC_NAMES = ["n_samples", "valid_fraction", "n_gaps", "n_clipped", "temp_mean", "photo_mean", "status"]

# The columns of hourly_features.csv, in order, as the data pipeline writes them
# with features.wrist: left (the wrist's file columns are left_zip and left_csv).
DIARY_COLUMNS = [
    "participant_id", "day", "hour_start", "hour_index", "visit", "kinesia_text", "kinesia_score",
    "kinesia_class", "tremor_score", "tremor_present", "freezing", "fall", "medication_taken",
    "medication", "dosage_mg", "wear_status", "left_zip", "left_csv", "hours_since_dose",
]
UPDRS_COLUMNS = ["updrs_off_total", "updrs_on_total", "levodopa_response", "updrs_days_since_exam",
                 "has_updrs_off", "has_updrs_on"]
DEMOGRAPHIC_COLUMNS = ["age", "sex", "handedness", "pd_subtype", "pd_dominantside", "hoehn_yahr",
                       "pd_yearssincediagnosis", "dbs", "age_group"]
FEATURES_CSV_COLUMNS = (DIARY_COLUMNS + [f"W_{name}" for name in FEATURE_NAMES + QC_NAMES]
                        + UPDRS_COLUMNS + DEMOGRAPHIC_COLUMNS)

# Typical (move_mean, still_fraction, tremor_mean) per class: off hours move least.
_CLASS_LEVELS = {"off": (0.005, 0.85, 0.010), "on": (0.012, 0.60, 0.012), "dyskinetic": (0.020, 0.40, 0.015)}
_KINESIA = {"off": [(-1, "Slight Akinesia"), (-2, "Discomforting Akinesia")],
            "on": [(0, "Good Kinesia")],
            "dyskinetic": [(1, "Slight Dyskinesia"), (2, "Discomforting Dyskinesia")]}
DOSE_HOURS = (8, 14, 20)


def make_features_table(n_participants=8, hours=24, seed=0):
    """A synthetic hourly_features.csv table: one row per participant-hour.

    Participants take turns through each demographic group, so with the
    default 8 every bias slice group has at least 2 of them. Every
    participant has both off and not-off hours.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_participants):
        person = _participant(i)
        classes = rng.choice(["off", "on", "dyskinetic"], size=hours, p=[0.35, 0.45, 0.2])
        classes[:2] = ["off", "on"]
        for h, label in enumerate(classes):
            rows.append({**_diary_hour(person, 15 + h, label, rng),
                         **_wrist_features(label, rng), **person["updrs"], **person["demographics"]})
    return pd.DataFrame(rows, columns=FEATURES_CSV_COLUMNS)


def make_dataset(spec, n_participants=2, hours=3, first_hour=8, seed=0):
    """A small synthetic Dataset with every input group, cut to spec.

    Each participant has `hours` diary hours from first_hour, a window every
    spec.seconds through them, one dose at the start and one check-in an hour later.
    """
    from scripts.core.data import Dataset, Windows
    from scripts.core.targets import build_labels

    rng = np.random.default_rng(seed)
    participants = [f"COPS-{i + 1}" for i in range(n_participants)]
    hour_rows, windows, events = [], {}, []
    for p in participants:
        for h in range(first_hour, first_hour + hours):
            score = int(rng.integers(-3, 4))
            hour_rows.append({"participant_id": p, "hour_index": h, "kinesia_text": "Good Kinesia",
                              "kinesia_score": float(score), "tremor_score": float(rng.integers(0, 3)),
                              "sex": "female"})
        start = first_hour * 3600
        n = hours * 3600 // spec.seconds
        windows[p] = Windows(time_s=start + spec.seconds * np.arange(n, dtype=np.int64),
                             worn=np.ones(n, dtype=bool),
                             signal=rng.normal(0, 0.01, (n, spec.axes, spec.samples)).astype(np.float32))
        events.append({"participant_id": p, "time_s": start + 60, "kind": "dose", "ledd": 100.0,
                       "source": "on_time", "kinesia_score": np.nan, "tremor_score": np.nan})
        events.append({"participant_id": p, "time_s": start + 3600 + 30, "kind": "check_in", "ledd": np.nan,
                       "source": None, "kinesia_score": 0.0, "tremor_score": 1.0})
    hours_table = pd.DataFrame(hour_rows)
    return Dataset(
        spec=spec,
        hours=hours_table[["participant_id", "hour_index", "sex"]],
        labels=build_labels(hours_table, ["state", "tremor"]),
        features=pd.DataFrame(rng.random((len(hours_table), len(FEATURE_NAMES))), columns=FEATURE_NAMES),
        windows=windows,
        events=pd.DataFrame(events),
        patients=pd.DataFrame({"participant_id": participants, "more_affected_side": True,
                               "dbs": [i % 2 == 0 for i in range(n_participants)]}),
    )


def write_features_csv(folder, **kwargs):
    """Write make_features_table(**kwargs) to folder/hourly_features.csv and return its path."""
    return write_table(make_features_table(**kwargs), folder / "hourly_features.csv")


def make_config(tmp_path, features_csv=None, **sections):
    """Config with every output under tmp_path.

    It reads features_csv, or a new synthetic table written under tmp_path.
    """
    if features_csv is None:
        features_csv = write_features_csv(tmp_path / "data")
    overrides = {"project_root": str(tmp_path), "paths": {"features_csv": str(features_csv)}}
    for key, value in sections.items():
        overrides.setdefault(key, {}).update(value)
    return load_config(overrides=overrides)


def _participant(i):
    age = [50, 60, 70][i % 3]
    off_total, on_total = 30 + 2 * i, 15 + i
    return {
        "participant_id": f"COPS-{i + 1}",
        "updrs": {"updrs_off_total": off_total, "updrs_on_total": on_total,
                  "levodopa_response": (off_total - on_total) / off_total,
                  "updrs_days_since_exam": 100 + 10 * i, "has_updrs_off": 1, "has_updrs_on": 1},
        "demographics": {
            "age": age, "sex": ["female", "male"][i % 2], "handedness": "right",
            "pd_subtype": ["akinetic-rigid", "equivalence-type"][i // 4 % 2],
            "pd_dominantside": ["right", "left"][i // 2 % 2], "hoehn_yahr": float(2 + i % 3),
            "pd_yearssincediagnosis": 10 + i, "dbs": ["no", "yes"][i // 2 % 2],
            "age_group": "<55" if age < 55 else "55-64" if age < 65 else "65+",
        },
        "visit": ["inpatient", "outpatient", "mixed"][i % 3],
    }


def _diary_hour(person, hour_index, label, rng):
    pid = person["participant_id"]
    day, hour = divmod(hour_index, 24)
    score, text = _KINESIA[label][rng.integers(len(_KINESIA[label]))]
    dosed = hour in DOSE_HOURS
    stem = f"{pid}_Day{day}_{hour}h-{hour + 1}h_leftWrist"
    return {
        "participant_id": pid, "day": day, "hour_start": hour, "hour_index": hour_index,
        "visit": person["visit"],
        "kinesia_text": text, "kinesia_score": float(score), "kinesia_class": label,
        "tremor_score": 0.0, "tremor_present": 0.0, "freezing": 0.0, "fall": 0.0,
        "medication_taken": int(dosed), "medication": "Levodopa / Carbidopa" if dosed else None,
        "dosage_mg": "100" if dosed else None, "wear_status": "Wearable Worn",
        "left_zip": f"{stem}.zip", "left_csv": f"{stem}.csv",
        "hours_since_dose": float(min(hour - d for d in (0,) + DOSE_HOURS if d <= hour)),
    }


def _wrist_features(label, rng):
    move, still, tremor = _CLASS_LEVELS[label]
    move_mean = move * rng.lognormal(0, 0.3)
    tremor_mean = tremor * rng.lognormal(0, 0.3)
    return {
        "W_move_mean": move_mean, "W_move_std": 2 * move_mean * rng.lognormal(0, 0.1),
        "W_move_p95": 4.5 * move_mean * rng.lognormal(0, 0.1),
        "W_still_fraction": float(np.clip(still + rng.normal(0, 0.08), 0, 1)),
        "W_tremor_mean": tremor_mean, "W_tremor_p95": 4 * tremor_mean * rng.lognormal(0, 0.1),
        "W_tremor_ratio": tremor_mean / move_mean,
        "W_n_samples": 360000.0, "W_valid_fraction": 1.0, "W_n_gaps": 0.0, "W_n_clipped": 0.0,
        "W_temp_mean": 30 + rng.normal(0, 1), "W_photo_mean": 20 + rng.normal(0, 5), "W_status": "ok",
    }
