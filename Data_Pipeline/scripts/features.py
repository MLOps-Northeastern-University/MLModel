"""Turn one hour of raw 100 Hz wrist accelerometry into a row of features.

Steps for each hourly file:
    1. Read X, Y, Z (in g), drop broken rows, clip impossible values
    2. Magnitude per row: sqrt(X^2 + Y^2 + Z^2)
    3. Band-pass filter the magnitude:
         0.1-3 Hz -> everyday movement (also removes gravity)
         4-8 Hz   -> tremor
    4. Absolute value, then average per 1-second epoch
    5. Summarise the hour: mean, variability, peak, time still, tremor ratio
"""

import io
import zipfile
from pathlib import PurePosixPath

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
from scipy.signal import butter, sosfiltfilt

from scripts.utils.logging_setup import get_logger

logger = get_logger("features")

FEATURE_NAMES = [
    "move_mean", "move_std", "move_p95", "still_fraction",
    "tremor_mean", "tremor_p95", "tremor_ratio",
]
QC_NAMES = ["n_samples", "valid_fraction", "n_gaps", "n_clipped", "temp_mean", "photo_mean"]


# Typed columns for the fast reader. Clock times are parsed as times while the
# file is read, so no separate text-to-time step is needed.
_ARROW_TYPES = {"Time": pa.time32("ms"), "X": pa.float64(), "Y": pa.float64(), "Z": pa.float64(),
                "Photo": pa.float64(), "Temp": pa.float64()}


def read_hour_csv(file_obj, sep=";"):
    """Read one hourly accelerometry CSV (columns Time;X;Y;Z;Photo;Temp).

    Most files are clean, so they're read with pyarrow, typed as they're read:
    about 20x faster than pandas' reader plus converting the times afterwards.
    Time then holds seconds after midnight. A file with a value pyarrow can't
    type (a broken row, an odd time) is read again by pandas, as text, and the
    feature step cleans it up.
    """
    data = file_obj.read()
    try:
        df = _read_typed(data, sep)
    except pa.ArrowInvalid:
        df = pd.read_csv(io.BytesIO(data), sep=sep)
    expected = {"Time", "X", "Y", "Z"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Accelerometry file missing columns {expected - set(df.columns)}")
    return df


def _read_typed(data, sep):
    """The fast reader: pyarrow with typed columns. Raises ArrowInvalid on any untypable value."""
    table = pa_csv.read_csv(
        io.BytesIO(data),
        parse_options=pa_csv.ParseOptions(delimiter=sep),
        convert_options=pa_csv.ConvertOptions(column_types=_ARROW_TYPES),
    )
    if "Time" not in table.column_names:
        return table.to_pandas()
    # time32[ms] -> milliseconds after midnight -> seconds; missing times become NaN.
    millis = table.column("Time").cast(pa.int32()).to_numpy(zero_copy_only=False)
    df = table.drop_columns(["Time"]).to_pandas()
    df.insert(0, "Time", pd.Series(millis, dtype="float64") / 1000)
    return df


def _time_to_seconds(series):
    """Convert 'HH:MM:SS.mmm' clock times to seconds after midnight.

    Times read by the fast reader are already seconds and are returned as they
    are. Text times have a fixed 12-character layout, so digits are read by
    position (about 20x faster than general parsing). Any value with a
    different layout falls back to pandas' general parser.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    arr = series.astype(str).to_numpy(dtype="U12")
    codes = arr.view(np.uint32).reshape(-1, 12).astype(np.int64)
    digit_pos = [0, 1, 3, 4, 6, 7, 9, 10, 11]
    well_formed = (
        (np.char.str_len(arr) == 12)
        & (codes[:, 2] == ord(":")) & (codes[:, 5] == ord(":")) & (codes[:, 8] == ord("."))
        & ((codes[:, digit_pos] >= ord("0")) & (codes[:, digit_pos] <= ord("9"))).all(axis=1)
    )
    d = codes - ord("0")
    seconds = ((d[:, 0] * 10 + d[:, 1]) * 3600 + (d[:, 3] * 10 + d[:, 4]) * 60
               + (d[:, 6] * 10 + d[:, 7]) + (d[:, 9] * 100 + d[:, 10] * 10 + d[:, 11]) / 1000)
    result = pd.Series(np.where(well_formed, seconds, np.nan), index=series.index)
    if not well_formed.all():
        odd = series[~well_formed]
        result[~well_formed] = pd.to_timedelta(odd, errors="coerce").dt.total_seconds()
    return result


def bandpass(signal, low, high, fs, order=4):
    """Zero-phase Butterworth band-pass filter."""
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)


def empty_features(n_samples=0, reason="no_data"):
    feats = {name: np.nan for name in FEATURE_NAMES}
    feats.update({name: np.nan for name in QC_NAMES})
    feats.update({"n_samples": n_samples, "valid_fraction": 0.0, "status": reason})
    return feats


def compute_hour_features(df, cfg):
    """Compute movement and tremor features for one hour of one wrist.

    Returns:
        dict: Features plus quality-control values and a 'status' field
        ('ok', 'insufficient_data').
    """
    fc = cfg["features"]
    fs = fc["sampling_rate_hz"]
    expected = fs * fc["expected_seconds_per_file"]

    for col in ["X", "Y", "Z", "Photo", "Temp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    valid = df.dropna(subset=["X", "Y", "Z"])
    n_valid = len(valid)
    if n_valid < fs * fc["min_valid_seconds"]:
        return empty_features(len(df), "insufficient_data")

    lo, hi = fc["accel_range_g"]
    xyz = valid[["X", "Y", "Z"]].to_numpy(dtype=float)
    n_clipped = int(((xyz < lo) | (xyz > hi)).any(axis=1).sum())
    xyz = np.clip(xyz, lo, hi)

    seconds = _time_to_seconds(valid["Time"])
    gaps = seconds.diff().dropna()
    n_gaps = int((gaps > fc["max_sample_gap_seconds"]).sum())

    magnitude = np.sqrt((xyz ** 2).sum(axis=1))
    order = fc["filter_order"]
    movement = np.abs(bandpass(magnitude, *fc["movement_band_hz"], fs=fs, order=order))
    tremor = np.abs(bandpass(magnitude, *fc["tremor_band_hz"], fs=fs, order=order))

    epoch = fs * fc["epoch_seconds"]
    n_epochs = len(magnitude) // epoch
    move_epochs = movement[: n_epochs * epoch].reshape(n_epochs, epoch).mean(axis=1)
    trem_epochs = tremor[: n_epochs * epoch].reshape(n_epochs, epoch).mean(axis=1)

    move_mean = float(move_epochs.mean())
    tremor_mean = float(trem_epochs.mean())
    return {
        "move_mean": move_mean,
        "move_std": float(move_epochs.std()),
        "move_p95": float(np.percentile(move_epochs, 95)),
        "still_fraction": float((move_epochs < fc["still_threshold_g"]).mean()),
        "tremor_mean": tremor_mean,
        "tremor_p95": float(np.percentile(trem_epochs, 95)),
        "tremor_ratio": tremor_mean / move_mean if move_mean > 0 else np.nan,
        "n_samples": int(len(df)),
        "valid_fraction": min(1.0, n_valid / expected),
        "n_gaps": n_gaps,
        "n_clipped": n_clipped,
        "temp_mean": float(valid["Temp"].mean()) if "Temp" in valid else np.nan,
        "photo_mean": float(valid["Photo"].mean()) if "Photo" in valid else np.nan,
        "status": "ok",
    }


class ParticipantArchive:
    """Read files from a participant zip without extracting it to disk.

    Hourly accelerometry files are stored as small zips inside the
    participant zip (nested), so each one is opened in memory.
    """

    def __init__(self, zip_path):
        self.path = zip_path
        self.zip = zipfile.ZipFile(zip_path)
        self.index = {PurePosixPath(n).name: n for n in self.zip.namelist() if not n.endswith("/")}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.zip.close()

    def has(self, name):
        return isinstance(name, str) and name in self.index

    def open_bytes(self, name):
        return io.BytesIO(self.zip.read(self.index[name]))

    def find(self, suffix):
        """Return the first file name ending with suffix, or None."""
        return next((n for n in self.index if n.endswith(suffix)), None)

    def read_accelerometry(self, zip_name=None, csv_name=None, sep=";"):
        """Read one hourly accelerometry file, zipped or plain."""
        if self.has(zip_name):
            with zipfile.ZipFile(self.open_bytes(zip_name)) as inner:
                csvs = [n for n in inner.namelist() if n.lower().endswith(".csv")]
                if not csvs:
                    raise ValueError(f"No CSV inside {zip_name}")
                with inner.open(csvs[0]) as f:
                    return read_hour_csv(f, sep)
        if self.has(csv_name):
            return read_hour_csv(self.open_bytes(csv_name), sep)
        raise FileNotFoundError(f"Accelerometry file not found: {zip_name or csv_name}")


def wrist_features(archive, zip_name, csv_name, cfg):
    """Features for one wrist and hour, never raising on bad files."""
    if pd.isna(zip_name) and pd.isna(csv_name):
        return empty_features(0, "no_file")
    try:
        df = archive.read_accelerometry(zip_name, csv_name, cfg["files"]["csv_separator"])
        return compute_hour_features(df, cfg)
    except FileNotFoundError:
        logger.warning("Missing accelerometry file %s", zip_name)
        return empty_features(0, "missing_file")
    except (zipfile.BadZipFile, ValueError, pd.errors.ParserError) as exc:
        logger.warning("Corrupt accelerometry file %s: %s", zip_name, exc)
        return empty_features(0, "corrupt_file")
