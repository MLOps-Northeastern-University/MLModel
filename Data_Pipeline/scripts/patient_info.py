"""Patient-level information: demographics and UPDRS-III exams.

These values are the same for every hour of a participant. They are used
for bias slicing and as optional patient-level features.
"""

import numpy as np
import pandas as pd

from scripts.utils.file_io import read_text_csv
from scripts.utils.logging_setup import get_logger

logger = get_logger("patient_info")


def load_updrs(source, sep=";"):
    """Read one UPDRS file (a single exam) and return its key values."""
    df = read_text_csv(source, sep=sep, na_values=["NaN", ""])
    if df.empty or "TotalScore" not in df.columns:
        raise ValueError("UPDRS file is empty or has no TotalScore column")
    row = df.iloc[0]
    return {
        "total": pd.to_numeric(row.get("TotalScore"), errors="coerce"),
        "days_since_exam": pd.to_numeric(row.get("DaysSinceExamination"), errors="coerce"),
        "dbs_state": pd.to_numeric(row.get("DBSState"), errors="coerce"),
    }


def summarise_updrs(off, on):
    """Combine OFF and ON exams. Missing exams become NaN, not errors."""
    off = off or {}
    on = on or {}
    off_total, on_total = off.get("total", np.nan), on.get("total", np.nan)
    response = (off_total - on_total) / off_total if off_total and not np.isnan(off_total) \
        and not np.isnan(on_total) else np.nan
    return {
        "updrs_off_total": off_total,
        "updrs_on_total": on_total,
        "levodopa_response": response,
        "updrs_days_since_exam": off.get("days_since_exam", on.get("days_since_exam", np.nan)),
        "has_updrs_off": int(bool(off)),
        "has_updrs_on": int(bool(on)),
    }


def load_demographics(path, cfg):
    """Read Demographics.csv and standardise its column names.

    The separator is detected automatically. Column names are lower-cased,
    spaces become underscores, and config demographics.column_map renames
    them to standard names (participant_id, sex, age, ...).
    """
    df = read_text_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns=cfg["demographics"]["column_map"])

    if "participant_id" not in df.columns:
        id_col = next(
            (c for c in df.columns if df[c].astype(str).str.startswith("COPS-").any()), None
        )
        if id_col is None:
            raise ValueError("Could not find the participant ID column in demographics")
        df = df.rename(columns={id_col: "participant_id"})
    df["participant_id"] = df["participant_id"].astype(str).str.strip()

    if "age" in df.columns:
        bias = cfg["bias"]
        df["age_group"] = pd.cut(
            pd.to_numeric(df["age"], errors="coerce"), bins=bias["age_bins"],
            labels=bias["age_labels"], right=False,
        ).astype(str)
    logger.info("Loaded demographics: %d participants, columns %s", len(df), list(df.columns))
    return df
