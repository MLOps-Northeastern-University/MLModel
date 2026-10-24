"""Pipeline steps shared by the Airflow DAG and the command-line runner.

Each function takes the config plus simple inputs (paths, dicts) and
returns simple outputs, so Airflow can pass results between tasks.
"""

from pathlib import Path

from scripts import acquire
from scripts.anomalies import DataAnomalyError, detect_anomalies, format_report
from scripts.alerts import notify
from scripts.bias import run_bias_analysis
from scripts.clean import clean_hours, drop_duplicate_hours
from scripts.join import attach_demographics, merge_participants
from scripts.patient_info import load_demographics
from scripts.process_patient import process_participant
from scripts.utils.file_io import read_json, read_table, write_json, write_table
from scripts.utils.logging_setup import get_logger
from scripts.validate import run_validation

logger = get_logger("steps")


def step_manifest(cfg, mode=None):
    return acquire.fetch_manifest(cfg, mode)


def step_demographics(cfg, manifest):
    """Download Demographics.csv and return its path (or None)."""
    entry = manifest.get("demographics")
    if not entry:
        logger.warning("No demographics file in manifest")
        return None
    return str(acquire.download_file(entry, cfg))


def step_participant(cfg, entry):
    """Download one participant zip, extract features, then delete the zip."""
    zip_path = acquire.download_file(entry, cfg)
    result = process_participant(zip_path, cfg)
    is_download = not entry.get("local_path")
    if is_download and cfg["data"]["delete_raw_after_processing"]:
        Path(zip_path).unlink(missing_ok=True)
        logger.info("Deleted raw zip %s", Path(zip_path).name)
    return result


class TooManyFailures(RuntimeError):
    """More participants failed than data.max_failed_fraction allows."""


def record_failures(cfg, failures, total, alert=None):
    """Save the failed participants, alert on them, and check the limit.

    Shared by the command line and the DAG, so both treat failures the same.
    Writes reports/failed_participants.json (an empty list when none failed).

    Returns:
        bool: True when more than data.max_failed_fraction of participants failed.
    """
    write_json(failures, cfg["resolved_paths"]["reports_dir"] / "failed_participants.json")
    if not failures:
        return False
    (alert or notify)(f"[COPS pipeline] {len(failures)} of {total} participants failed",
                      "\n".join(f"{f['participant_id']}: {f['error']}" for f in failures), cfg)
    limit = cfg["data"]["max_failed_fraction"]
    share = len(failures) / total
    if share > limit:
        logger.error("%d of %d participants failed (%.0f%%, limit %.0f%%)",
                     len(failures), total, share * 100, limit * 100)
        return True
    return False


def step_check_failures(cfg, manifest, results):
    """After every participant has finished or failed (Airflow), keep the ones that worked.

    A participant that still fails after its retries leaves no result. The
    run carries on without it, as the command line does, unless more than
    data.max_failed_fraction failed.

    Returns:
        list: The results of the participants that worked.
    """
    results = [r for r in results if r]
    done = {r["participant_id"] for r in results}
    failures = [
        {"participant_id": Path(p["name"]).stem,
         "error": "failed after its retries; see the process_participant task log"}
        for p in manifest["participants"] if Path(p["name"]).stem not in done
    ]
    if record_failures(cfg, failures, len(manifest["participants"])) or not results:
        raise TooManyFailures(f"{len(failures)} of {len(manifest['participants'])} participants failed")
    return results


def step_merge(cfg, results, demographics_path):
    merged, issues = merge_participants(results)
    demographics = load_demographics(demographics_path, cfg) if demographics_path else None
    merged = attach_demographics(merged, demographics)
    processed = cfg["resolved_paths"]["processed_dir"]
    write_json(issues, processed / "label_issues.json")
    return str(write_table(merged, processed / "hourly_all.csv"))


def step_clean(cfg, merged_path):
    clean, summary = clean_hours(read_table(merged_path), cfg)
    clean = drop_duplicate_hours(clean)
    write_json(summary, cfg["resolved_paths"]["reports_dir"] / "cleaning_summary.json")
    return str(write_table(clean, cfg["resolved_paths"]["processed_dir"] / "hourly_features.csv"))


def step_validate(cfg, features_path, run_id="manual"):
    violations = run_validation(read_table(features_path), cfg, run_id)
    return str(cfg["resolved_paths"]["reports_dir"] / "validation_report.json"), len(violations)


def step_anomalies(cfg, features_path):
    """Detect anomalies, alert on any, and stop the pipeline on critical ones."""
    df = read_table(features_path)
    issues = read_json(cfg["resolved_paths"]["processed_dir"] / "label_issues.json")
    violations = read_json(cfg["resolved_paths"]["reports_dir"] / "validation_report.json")
    report = detect_anomalies(df, cfg, issues, violations)
    text = format_report(report)
    if report["critical"]:
        notify("[COPS pipeline] Critical data anomalies", text, cfg)
        raise DataAnomalyError(text)
    if report["warnings"]:
        notify("[COPS pipeline] Data anomaly warnings", text, cfg)
    return report


def step_bias(cfg, features_path):
    run_bias_analysis(read_table(features_path), cfg)
    return str(cfg["resolved_paths"]["reports_dir"] / "bias_report.md")
