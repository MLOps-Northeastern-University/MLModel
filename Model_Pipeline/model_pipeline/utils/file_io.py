"""Small file helpers shared by the pipeline steps."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Columns that hold text even when a file's values all look like numbers
# (dosage "50" in one file, "50;50" in another), so reading never changes their type.
TEXT_COLUMNS = {"dosage_mg": str}


def write_table(df, path):
    """Write a DataFrame to CSV, creating the parent folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def read_table(path):
    """Read a CSV written by write_table, or the data pipeline's hourly_features.csv."""
    return pd.read_csv(path, dtype=TEXT_COLUMNS)


def write_json(obj, path):
    """Write an object to JSON, converting numpy types."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_to_builtin)
    return path


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _to_builtin(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(f"Cannot serialise {type(value)}")
