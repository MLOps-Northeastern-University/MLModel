"""Reading raw files: participant zips, hourly accelerometer CSVs, demographics."""

import pandas as pd
import pytest

from scripts.read import Archive, participant_id, read_demographics, read_hour_csv
from tests.helpers import csv_bytes, make_config, make_demographics, make_participant_zip, signal


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path)


def test_participant_id_comes_from_the_file_name():
    assert participant_id("/x/COPS-7.zip") == "COPS-7"
    with pytest.raises(ValueError):
        participant_id("notes.zip")


def test_an_archive_gives_the_diary_and_each_hour(tmp_path):
    with Archive(make_participant_zip(tmp_path, "COPS-1")) as a:
        assert len(a.diary()) == 7
        df = a.accelerometer("COPS-1_Day0_9h-10h_leftWrist.zip", None)
        assert list(df.columns[:4]) == ["Time", "X", "Y", "Z"]
        assert df["Time"].iloc[0] == pytest.approx(9 * 3600 + 0.01)  # seconds after midnight
        with pytest.raises(FileNotFoundError):
            a.accelerometer("COPS-1_Day9_1h-2h_leftWrist.zip", None)


def test_a_plain_csv_hour_is_read_too(tmp_path):
    import zipfile
    path = tmp_path / "COPS-2.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("COPS-2/hour.csv", csv_bytes(signal(hour=3)))
    with Archive(path) as a:
        assert len(a.accelerometer(None, "hour.csv")) == 12000
        with pytest.raises(FileNotFoundError, match="symptom diary"):
            a.diary()


def test_a_broken_row_falls_back_and_becomes_nan():
    df = signal(seconds=1).astype({"X": object})
    df.loc[3, "X"] = "oops"
    df.loc[5, "Time"] = "not a time"
    out = read_hour_csv(csv_bytes(df))
    assert pd.isna(out.loc[3, "X"]) and pd.isna(out.loc[5, "Time"])
    assert out["Time"].iloc[0] == pytest.approx(10 * 3600 + 0.01)


def test_a_file_without_accelerometer_columns_is_refused():
    with pytest.raises(ValueError, match="missing columns"):
        read_hour_csv(b"Time;Temp\n10:00:00.000;30\n")


def test_demographics_get_standard_names_and_an_age_group(cfg, tmp_path):
    demo = read_demographics(make_demographics(tmp_path, ["COPS-1", "COPS-2"], ["right", "left"]), cfg)
    assert {"participant_id", "pd_dominantside", "hoehn_yahr", "pd_subtype", "dbs", "sex"} <= set(demo.columns)
    assert demo["age_group"].tolist() == ["<55", "55-64"]
    assert demo["pd_dominantside"].tolist() == ["right", "left"]


def test_demographics_without_an_id_column_are_refused(cfg, tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("age;sex\n50;male\n")
    with pytest.raises(ValueError, match="participant ID"):
        read_demographics(path, cfg)
