"""Small file helpers shared by the pipeline steps."""

import io
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
    """Read a CSV written by write_table."""
    return pd.read_csv(path, dtype=TEXT_COLUMNS)


def read_text_csv(source, **kwargs):
    """pd.read_csv for the dataset's own text files (diaries, UPDRS, demographics).

    They're usually UTF-8, sometimes with the byte-order mark Excel adds, but a
    file saved on Windows can be cp1252 (German text: "Benserazid", umlauts).
    UTF-8 is tried first, then cp1252, then Latin-1, which accepts any bytes.

    Args:
        source: A path or a binary file object.
        **kwargs: Passed on to pd.read_csv (sep, na_values, ...).
    """
    data = source.read() if hasattr(source, "read") else Path(source).read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.StringIO(text), **kwargs)


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
