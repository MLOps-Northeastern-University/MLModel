import numpy as np
import pandas as pd
import pytest

from scripts.core.data import WindowSpec
from scripts.core.errors import SchemaError
from scripts.data.loaders import data_version, load_dataset
from scripts.utils.file_io import read_table, write_json, write_table
from tests.helpers import make_config, write_processed

SPEC = WindowSpec(1, 10, 30, 3)


@pytest.fixture
def cfg(tmp_path):
    cfg = make_config(tmp_path)
    write_processed(cfg["resolved_paths"]["processed_data"], SPEC, n_participants=3)
    return cfg


def test_the_processed_folder_loads_as_a_checked_dataset(cfg):
    data = load_dataset(cfg)
    assert data.participants == ("COPS-1", "COPS-2", "COPS-3") and data.spec == SPEC
    assert list(data.labels.columns) == ["state_now", "state_next_hour", "tremor_now", "tremor_next_hour"]
    assert list(data.features.columns) == [f"W_{n}" for n in cfg["data"]["features"]]
    assert not any(c.startswith("W_") for c in data.hours.columns)
    assert data.patients["dbs"].tolist() == [True, False, True]


def test_windows_are_memory_mapped_not_copied(cfg):
    signal = load_dataset(cfg).windows["COPS-1"].signal
    assert isinstance(signal, np.memmap) and signal.dtype == np.float32 and signal.shape[1:] == (3, 300)


def test_labels_come_from_the_diary_answers(cfg):
    data = load_dataset(cfg)
    np.testing.assert_array_equal(data.labels["state_now"], data.hours["kinesia_score"] + 3)


def test_missing_data_says_how_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="dvc pull"):
        load_dataset(make_config(tmp_path))


def test_a_participant_without_windows_is_reported(cfg):
    folder = cfg["resolved_paths"]["processed_data"] / "windows" / "COPS-2"
    for f in folder.iterdir():
        f.unlink()
    folder.rmdir()
    with pytest.raises(FileNotFoundError, match="COPS-2"):
        load_dataset(cfg)


def test_windows_cut_with_another_spec_are_refused(cfg):
    path = cfg["resolved_paths"]["processed_data"] / "dataset_meta.json"
    write_json({"spec_version": 2, "seconds": 10, "rate_hz": 30, "axes": 3}, path)
    with pytest.raises(SchemaError, match="spec"):
        load_dataset(cfg)


def test_a_blank_dbs_is_refused(cfg):
    path = cfg["resolved_paths"]["processed_data"] / "patients.csv"
    patients = read_table(path)
    patients["dbs"] = patients["dbs"].astype(object)
    patients.loc[0, "dbs"] = None
    write_table(patients, path)
    with pytest.raises(SchemaError, match="dbs"):
        load_dataset(cfg)


def test_the_data_version_follows_the_dvc_lock(cfg):
    lock = cfg["resolved_paths"]["dvc_lock"]
    assert data_version(cfg) is None

    def write_lock(md5):
        lock.write_text(f"schema: '2.0'\nstages:\n  build_dataset:\n    outs:\n"
                        f"    - path: data/processed/hours.csv\n      md5: {md5}\n")
    write_lock("aaa")
    first = data_version(cfg)
    assert len(first) == 12 and data_version(cfg) == first
    write_lock("bbb")
    assert data_version(cfg) != first


def test_the_hours_index_is_shared_by_every_table(cfg):
    data = load_dataset(cfg)
    assert data.labels.index.equals(data.hours.index) and data.features.index.equals(data.hours.index)
    assert isinstance(data.hours.index, pd.RangeIndex)
