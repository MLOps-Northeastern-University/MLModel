"""The synthetic table is usable for every owner's tests: real columns, both classes, every bias group."""

import pandas as pd

from model_pipeline.utils.file_io import read_table
from tests.helpers import FEATURES_CSV_COLUMNS, make_config, make_features_table, write_features_csv


def test_the_table_has_the_data_pipelines_one_wrist_columns():
    df = make_features_table()
    assert list(df.columns) == FEATURES_CSV_COLUMNS
    assert not any(c.startswith(("L_", "R_", "asym_")) for c in df.columns)
    assert df["W_status"].eq("ok").all()


def test_every_participant_has_off_and_not_off_hours():
    df = make_features_table()
    off = df["kinesia_class"].eq("off")
    assert off.groupby(df["participant_id"]).agg(["any", "all"]).eq([True, False]).all().all()


def test_every_bias_group_has_at_least_two_participants(tmp_path):
    cfg = make_config(tmp_path)
    df = make_features_table()
    for column in cfg["bias"]["slice_columns"]:
        assert df.groupby(column)["participant_id"].nunique().min() >= 2, column


def test_off_hours_move_less_so_a_model_has_something_to_learn():
    df = make_features_table()
    means = df.groupby("kinesia_class")["W_move_mean"].mean()
    assert means["off"] < means["on"] < means["dyskinetic"]


def test_the_same_seed_gives_the_same_table():
    pd.testing.assert_frame_equal(make_features_table(seed=3), make_features_table(seed=3))


def test_the_table_reads_back_like_the_real_file(tmp_path):
    path = write_features_csv(tmp_path, n_participants=4, hours=10)
    df = read_table(path)
    assert list(df.columns) == FEATURES_CSV_COLUMNS
    assert len(df) == 40
    assert df["dosage_mg"].dropna().eq("100").all()
