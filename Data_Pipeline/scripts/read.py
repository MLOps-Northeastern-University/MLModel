"""Read the raw COPS files: participant zips, diaries, hourly accelerometer files, demographics.

A participant zip holds the symptom diary and one small zip per wrist per hour,
each with one CSV (Time;X;Y;Z;Photo;Temp at 100 Hz). Nothing is unzipped to disk.
"""

import io
import re
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv

from scripts.utils import get_logger, read_text_csv

logger = get_logger("read")

ACCEL_COLUMNS = ["Time", "X", "Y", "Z"]
_ARROW_TYPES = {"Time": pa.time32("ms"), "X": pa.float64(), "Y": pa.float64(), "Z": pa.float64(),
                "Photo": pa.float64(), "Temp": pa.float64()}


def participant_id(path):
    match = re.search(r"(COPS-\d+)", Path(path).name)
    if not match:
        raise ValueError(f"Cannot read a participant ID from {path}")
    return match.group(1)


class Archive:
    """A participant zip, read in memory."""

    def __init__(self, path):
        self.zip = zipfile.ZipFile(path)
        self.index = {PurePosixPath(n).name: n for n in self.zip.namelist() if not n.endswith("/")}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.zip.close()

    def has(self, name):
        return isinstance(name, str) and name in self.index

    def open(self, name):
        return io.BytesIO(self.zip.read(self.index[name]))

    def find(self, suffix):
        """The first file name ending with suffix, or None."""
        return next((n for n in self.index if n.endswith(suffix)), None)

    def diary(self, sep=";"):
        name = self.find("_symptomdiary.csv")
        if name is None:
            raise FileNotFoundError("No symptom diary in the zip")
        return read_text_csv(self.open(name), sep=sep, na_values=["NaN", ""])

    def accelerometer(self, zip_name, csv_name, sep=";"):
        """One hour of one wrist, from its inner zip or a plain CSV."""
        if self.has(zip_name):
            with zipfile.ZipFile(self.open(zip_name)) as inner:
                csvs = [n for n in inner.namelist() if n.lower().endswith(".csv")]
                if not csvs:
                    raise ValueError(f"No CSV inside {zip_name}")
                return read_hour_csv(inner.read(csvs[0]), sep)
        if self.has(csv_name):
            return read_hour_csv(self.open(csv_name).read(), sep)
        raise FileNotFoundError(f"Accelerometer file not found: {zip_name}")


def read_hour_csv(data, sep=";"):
    """An hourly accelerometer CSV as numbers, with Time in seconds after midnight.

    Most files are clean and are read by pyarrow, typed as they're read (about 20x
    faster than pandas). A file with a value pyarrow can't type is read again by
    pandas; anything unreadable becomes NaN.
    """
    try:
        table = pa_csv.read_csv(io.BytesIO(data), parse_options=pa_csv.ParseOptions(delimiter=sep),
                                convert_options=pa_csv.ConvertOptions(column_types=_ARROW_TYPES))
        millis = table.column("Time").cast(pa.int32()).to_numpy(zero_copy_only=False)
        df = table.drop_columns(["Time"]).to_pandas()
        df.insert(0, "Time", pd.Series(millis, dtype="float64") / 1000)
    except (pa.ArrowInvalid, KeyError):
        df = pd.read_csv(io.BytesIO(data), sep=sep, dtype=str)
        if "Time" in df:
            df["Time"] = pd.to_timedelta(df["Time"], errors="coerce").dt.total_seconds()
        for col in df.columns.drop("Time", errors="ignore"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    missing = set(ACCEL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Accelerometer file missing columns {sorted(missing)}")
    return df


def read_demographics(path, cfg):
    """Demographics.csv with standard column names (config demographics.column_map) and an age_group."""
    demo = cfg["demographics"]
    df = read_text_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns=demo["column_map"])
    if "participant_id" not in df.columns:
        raise ValueError("Demographics has no participant ID column")
    df["participant_id"] = df["participant_id"].astype(str).str.strip()
    for col in df.columns[df.dtypes == object]:
        df[col] = df[col].str.strip()
    if "age" in df.columns:
        df["age_group"] = pd.cut(pd.to_numeric(df["age"], errors="coerce"), bins=demo["age_bins"],
                                 labels=demo["age_labels"], right=False).astype(str)
    logger.info("Demographics: %d participants", len(df))
    return df
