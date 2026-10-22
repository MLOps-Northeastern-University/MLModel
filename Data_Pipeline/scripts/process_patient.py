"""Process one participant zip into a table of hourly features and labels.

This is the unit of parallel work in the DAG: each participant is
independent, so Airflow runs several of these at the same time.
"""

import re
from pathlib import Path

import pandas as pd

from scripts.diary import load_diary, parse_diary
from scripts.features import ParticipantArchive, combine_wrists, wrist_features
from scripts.patient_info import load_updrs, summarise_updrs
from scripts.utils.file_io import write_json, write_table
from scripts.utils.logging_setup import get_logger

logger = get_logger("process_patient")


def participant_id_from_path(zip_path):
    match = re.search(r"(COPS-\d+)", Path(zip_path).name)
    if not match:
        raise ValueError(f"Cannot read participant ID from {zip_path}")
    return match.group(1)


def process_participant(zip_path, cfg):
    """Build the hourly feature table for one participant.

    Only diary hours that point to accelerometry files are processed, so
    hours with no sensor data are skipped cheaply. Filtering by label,
    wear status and quality happens later in clean.py.

    Returns:
        dict: Paths to the participant's feature table and issue report.
    """
    pid = participant_id_from_path(zip_path)
    sep = cfg["files"]["csv_separator"]
    logger.info("Processing %s", pid)

    with ParticipantArchive(zip_path) as archive:
        diary_name = archive.find("_symptomdiary.csv")
        if diary_name is None:
            raise FileNotFoundError(f"No symptom diary in {zip_path}")
        diary, issues = parse_diary(load_diary(archive.open_bytes(diary_name), sep), cfg)

        updrs = {}
        for state in ["OFF", "ON"]:
            name = archive.find(f"_UPDRS_{state}.csv")
            if name is None:
                logger.warning("%s has no UPDRS %s file", pid, state)
                issues.append({"participant_id": pid, "field": f"updrs_{state.lower()}",
                               "issue": "file_missing"})
                continue
            try:
                updrs[state] = load_updrs(archive.open_bytes(name), sep)
            except ValueError as exc:
                logger.warning("%s UPDRS %s unreadable: %s", pid, state, exc)
                issues.append({"participant_id": pid, "field": f"updrs_{state.lower()}",
                               "issue": "file_unreadable"})

        rows = []
        has_files = diary[["left_zip", "left_csv", "right_zip", "right_csv"]].notna().any(axis=1)
        hours = diary[has_files]
        for n, (_, hour) in enumerate(hours.iterrows(), start=1):
            left = wrist_features(archive, hour["left_zip"], hour["left_csv"], cfg)
            right = wrist_features(archive, hour["right_zip"], hour["right_csv"], cfg)
            rows.append({"hour_index": hour["hour_index"], **combine_wrists(left, right)})
            if n % 20 == 0 or n == len(hours):
                logger.info("%s: %d/%d hours done", pid, n, len(hours))

    features = pd.DataFrame(rows)
    table = diary.merge(features, on="hour_index", how="left") if len(features) else diary
    for key, value in summarise_updrs(updrs.get("OFF"), updrs.get("ON")).items():
        table[key] = value

    interim = cfg["resolved_paths"]["interim_dir"]
    table_path = write_table(table, interim / f"{pid}_hourly.csv")
    issues_path = write_json(issues, interim / f"{pid}_issues.json")
    logger.info("%s done: %d diary hours, %d with sensor features", pid, len(table), len(features))
    return {"participant_id": pid, "table": str(table_path), "issues": str(issues_path)}
