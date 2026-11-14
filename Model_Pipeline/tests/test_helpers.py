"""The synthetic data is shaped like the data pipeline's output."""

import numpy as np

from scripts.core.data import GROUPS, WindowSpec
from scripts.utils.file_io import read_json, read_table
from tests.helpers import FEATURE_NAMES, make_dataset, write_processed

SPEC = WindowSpec(1, 10, 30, 3)


def test_the_dataset_passes_every_check():
    make_dataset(SPEC).check(GROUPS, SPEC)


def test_the_same_seed_gives_the_same_data():
    a, b = make_dataset(SPEC, seed=3), make_dataset(SPEC, seed=3)
    assert a.labels.equals(b.labels)
    np.testing.assert_array_equal(a.windows["COPS-1"].signal, b.windows["COPS-1"].signal)


def test_the_processed_folder_has_the_data_pipelines_files(tmp_path):
    write_processed(tmp_path, SPEC, n_participants=3)
    hours = read_table(tmp_path / "hours.csv")
    assert {"participant_id", "hour_index", "kinesia_text", "kinesia_score", "tremor_score"} <= set(hours.columns)
    assert [f"W_{n}" for n in FEATURE_NAMES] == [c for c in hours.columns if c.startswith("W_")]
    assert read_json(tmp_path / "dataset_meta.json")["spec_version"] == 1
    assert np.load(tmp_path / "windows" / "COPS-3" / "signal.npy").shape[1:] == (3, 300)
