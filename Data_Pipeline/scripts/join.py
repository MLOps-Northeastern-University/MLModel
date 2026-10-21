"""Merge all participants and attach demographics."""

import json

import pandas as pd

from scripts.utils.logging_setup import get_logger

logger = get_logger("join")


def merge_participants(results):
    """Concatenate per-participant tables and collect their issues.

    Args:
        results: List of dicts returned by process_participant.
    """
    tables, issues = [], []
    for res in results:
        tables.append(pd.read_csv(res["table"]))
        with open(res["issues"], encoding="utf-8") as f:
            issues.extend(json.load(f))
    if not tables:
        raise ValueError("No participant tables to merge")
    merged = pd.concat(tables, ignore_index=True)
    logger.info("Merged %d participants into %d hourly rows", len(tables), len(merged))
    return merged, issues


def attach_demographics(df, demographics):
    """Left-join demographics on participant_id; warn about missing people."""
    if demographics is None:
        logger.warning("No demographics available; bias slices will be limited")
        return df
    missing = set(df["participant_id"]) - set(demographics["participant_id"])
    if missing:
        logger.warning("No demographics for: %s", sorted(missing))
    overlap = [c for c in demographics.columns if c in df.columns and c != "participant_id"]
    demo = demographics.drop(columns=overlap)
    return df.merge(demo, on="participant_id", how="left")
