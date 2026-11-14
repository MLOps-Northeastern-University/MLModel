import json

import numpy as np
import pandas as pd
import pytest

from scripts.core.data import GROUPS, WindowSpec
from scripts.core.errors import ConfigError, SchemaError
from scripts.data.loaders import WindowStore, load_dataset
from tests.helpers import FEATURE_NAMES, make_config

SPEC = WindowSpec(version=1, seconds=10, rate_hz=30, axes=3)
PEOPLE = ["COPS-1", "COPS-2", "COPS-3"]
HOURS = range(8, 11)


def write_files(root, people=PEOPLE, manifest=True, windows_for=None):
    """The data pipeline's outputs, synthetic, in its layout. Returns the config paths."""
    rng = np.random.default_rng(0)
    rows = [{"participant_id": p, "hour_index": h, "kinesia_text": "Good Kinesia", "kinesia_score": 0.0,
             "tremor_score": float(h % 3), "sex": "female"} for p in people for h in HOURS]
    rows[0].update(kinesia_text="Sleep", kinesia_score=np.nan)
    pd.DataFrame(rows[::-1]).to_csv(root / "hours.csv", index=False)   # unsorted, as a pipeline may write it
    features = pd.DataFrame([{"participant_id": r["participant_id"], "hour_index": r["hour_index"],
                              **{f"W_{n}": rng.random() for n in FEATURE_NAMES}} for r in rows[1:]])
    features.to_csv(root / "features.csv", index=False)   # the Sleep hour has no features
    folder = root / "windows"
    folder.mkdir()
    for p in windows_for if windows_for is not None else people:
        n = len(HOURS) * 360
        (folder / p).mkdir()
        np.save(folder / p / "signal.npy", rng.normal(0, 0.01, (n, 3, 300)).astype(np.float32))
        np.save(folder / p / "time_s.npy", HOURS[0] * 3600 + 10 * np.arange(n, dtype=np.int64))
        np.save(folder / p / "worn.npy", np.ones(n, dtype=bool))
    if manifest:
        (folder / "manifest.json").write_text(json.dumps({"spec_version": 1, "seconds": 10, "rate_hz": 30, "axes": 3}))
    events = [{"participant_id": p, "time_s": 8 * 3600 + 60, "kind": "dose", "ledd": 100.0, "source": "on_time",
               "kinesia_score": np.nan, "tremor_score": np.nan} for p in people]
    pd.DataFrame(events).to_csv(root / "events.csv", index=False)
    pd.DataFrame({"participant_id": people, "more_affected_side": True, "dbs": False}).to_csv(
        root / "patients.csv", index=False)
    return {"hours_csv": str(root / "hours.csv"), "features_csv": str(root / "features.csv"),
            "windows_folder": str(folder), "events_csv": str(root / "events.csv"),
            "patients_csv": str(root / "patients.csv")}


@pytest.fixture
def cfg(tmp_path):
    paths = write_files(tmp_path)
    return make_config(tmp_path, features_csv=paths["features_csv"], paths=paths)


def test_hours_and_labels_are_always_read_and_nothing_else(cfg):
    data = load_dataset(cfg)
    assert data.participants == tuple(PEOPLE) and len(data.hours) == 9
    assert data.windows is None and data.events is None and data.features is None
    assert {"sleep_now", "state_now", "tremor_next_hour"} <= set(data.labels.columns)
    assert data.labels["sleep_now"].iloc[0] == 1 and np.isnan(data.labels["state_now"].iloc[0])   # the Sleep hour


def test_hours_come_sorted_by_participant_and_hour(cfg):
    hours = load_dataset(cfg).hours
    assert list(hours.index) == list(range(9))
    assert hours[["participant_id", "hour_index"]].equals(hours.sort_values(["participant_id", "hour_index"])
                                                          [["participant_id", "hour_index"]])


def test_every_group_loads_and_passes_the_schema_checks(cfg):
    data = load_dataset(cfg, groups=GROUPS)
    data.check(GROUPS, SPEC)
    assert data.spec == SPEC


def test_windows_are_memory_mapped_and_read_on_first_use(cfg):
    store = load_dataset(cfg, groups=("windows",)).windows
    assert isinstance(store, WindowStore) and store._cache == {}
    w = store["COPS-2"]
    assert isinstance(w.signal, np.memmap) and w.signal.shape == (1080, 3, 300)
    assert set(store) == set(PEOPLE) and len(store) == 3


def test_features_are_matched_by_hour_without_the_prefix(cfg):
    features = load_dataset(cfg, groups=("features",)).features
    assert list(features.columns) == FEATURE_NAMES
    assert features.iloc[0].isna().all() and features.iloc[1:].notna().all().all()   # no features for the Sleep hour


def test_a_subset_of_participants(cfg):
    data = load_dataset(cfg, groups=GROUPS, participants=["COPS-3"])
    data.check(GROUPS, SPEC)
    assert data.participants == ("COPS-3",) and set(data.windows) == {"COPS-3"}
    assert set(data.events.participant_id) == set(data.patients.participant_id) == {"COPS-3"}
    with pytest.raises(SchemaError, match="aren't in the hours table"):
        load_dataset(cfg, participants=["COPS-9"])


def test_the_spec_comes_from_the_windows_own_manifest(tmp_path):
    paths = write_files(tmp_path)
    manifest = tmp_path / "windows" / "manifest.json"
    manifest.write_text(json.dumps({"spec_version": 2, "seconds": 10, "rate_hz": 24, "axes": 3}))
    data = load_dataset(make_config(tmp_path, features_csv=paths["features_csv"], paths=paths), groups=("windows",))
    assert data.spec.version == 2
    with pytest.raises(SchemaError, match="cut with spec v2"):
        data.check(("windows",), SPEC)


def test_windows_without_a_manifest_are_refused(tmp_path):
    paths = write_files(tmp_path, manifest=False)
    with pytest.raises(SchemaError, match="no manifest.json"):
        load_dataset(make_config(tmp_path, features_csv=paths["features_csv"], paths=paths), groups=("windows",))


def test_a_participant_with_no_windows_fails_the_check(tmp_path):
    paths = write_files(tmp_path, windows_for=PEOPLE[:2])
    data = load_dataset(make_config(tmp_path, features_csv=paths["features_csv"], paths=paths), groups=("windows",))
    with pytest.raises(SchemaError, match="have none"):
        data.check(("windows",), SPEC)


def test_a_missing_file_or_unknown_group_is_reported(cfg, tmp_path):
    (tmp_path / "events.csv").unlink()
    with pytest.raises(SchemaError, match="events_csv not found.*dvc pull"):
        load_dataset(cfg, groups=("events",))
    with pytest.raises(ConfigError, match="unknown input group"):
        load_dataset(cfg, groups=("medication",))


def test_duplicate_feature_rows_and_a_bad_window_spec_are_schema_errors(tmp_path):
    paths = write_files(tmp_path)
    features = pd.read_csv(paths["features_csv"])
    pd.concat([features, features.iloc[:1]]).to_csv(paths["features_csv"], index=False)
    cfg = make_config(tmp_path, features_csv=paths["features_csv"], paths=paths)
    with pytest.raises(SchemaError, match="twice"):
        load_dataset(cfg, groups=("features",))
    (tmp_path / "windows" / "manifest.json").write_text(json.dumps({"spec_version": 1, "seconds": 0, "rate_hz": 30,
                                                                    "axes": 3}))
    with pytest.raises(SchemaError, match="seconds"):
        load_dataset(cfg, groups=("windows",))
