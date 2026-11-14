"""Small synthetic data for tests, shaped like the data pipeline's output.

Nothing here comes from the real dataset. make_dataset builds a Dataset in memory;
write_processed writes one to disk as the data pipeline's data/processed/ folder,
so the loader can be tested without the real files.
"""

import numpy as np
import pandas as pd

from scripts.utils.config import load_config
from scripts.utils.file_io import write_json, write_table

FEATURE_NAMES = ["move_mean", "move_std", "move_p95", "still_fraction", "tremor_mean", "tremor_p95", "tremor_ratio"]


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
        hours=hours_table,
        labels=build_labels(hours_table, ["state", "tremor"]),
        features=pd.DataFrame(rng.random((len(hours_table), len(FEATURE_NAMES))), columns=FEATURE_NAMES),
        windows=windows,
        events=pd.DataFrame(events),
        patients=pd.DataFrame({"participant_id": participants, "more_affected_side": True,
                               "dbs": [i % 2 == 0 for i in range(n_participants)]}),
    )


def write_processed(folder, spec, **kwargs):
    """Write make_dataset(spec, **kwargs) as the data pipeline's processed folder; return the Dataset."""
    data = make_dataset(spec, **kwargs)
    hours = pd.concat([data.hours, data.features.add_prefix("W_")], axis=1)
    write_table(hours, folder / "hours.csv")
    write_table(data.events, folder / "events.csv")
    write_table(data.patients, folder / "patients.csv")
    for pid, w in data.windows.items():
        (folder / "windows" / pid).mkdir(parents=True)
        for name in ("time_s", "worn", "signal"):
            np.save(folder / "windows" / pid / f"{name}.npy", getattr(w, name))
    write_json({"spec_version": spec.version, "seconds": spec.seconds, "rate_hz": spec.rate_hz,
                "axes": spec.axes, "participants": len(data.windows)}, folder / "dataset_meta.json")
    return data


def make_config(tmp_path, **sections):
    """Config with every output under tmp_path, reading data from tmp_path/processed."""
    overrides = {"project_root": str(tmp_path),
                 "paths": {"processed_data": "processed", "dvc_lock": "dvc.lock"}}
    for key, value in sections.items():
        overrides.setdefault(key, {}).update(value)
    return load_config(overrides=overrides)
