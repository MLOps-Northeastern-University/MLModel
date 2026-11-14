"""The pipeline: patients, then each participant, then the dataset, then the checks.

The command line and the Airflow DAG run these same functions:

    python -m scripts.pipeline --mode sample

Zips already in data/raw/ are used instead of downloading them again.

Outputs (data/processed/), the input groups of Model_Pipeline's core/data.py:
    hours.csv          every diary hour: labels, bias slices, the 7 hourly features, drop_reason
    events.csv         every dose, with its LEDD
    patients.csv       one row per participant: wrist, more_affected_side, dbs, orientation fixes
    windows/<pid>/     time_s.npy, worn.npy, signal.npy (n, 3, 300) float32
    dataset_meta.json  the window spec and counts
"""

import argparse
import shutil
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import acquire, checks, sensor
from scripts.diary import NOT_WORN, SLEEP, parse_diary
from scripts.ledd import LeddTable, dose_events
from scripts.read import Archive, participant_id, read_demographics
from scripts.utils import get_logger, load_config, notify, read_json, read_table, write_json, write_table

logger = get_logger("pipeline")

SECONDS_PER_DAY, SECONDS_PER_HOUR = 86400, 3600
FILE_COLUMNS = [f"{side}_{kind}" for side in ("left", "right") for kind in ("zip", "csv")]
FIX_COLUMNS = ["participant_id", "wrist", "more_affected_side", "turned_round_share", "wear_bouts", "mirrored"]
PATIENT_COLUMNS = ["participant_id", "wrist", "more_affected_side", "dbs", "turned_round_share", "wear_bouts",
                   "mirrored"]
BOOL_TEXT = {"yes": True, "no": False, "true": True, "false": False, "1": True, "0": False}


class TooManyFailures(RuntimeError):
    """More participants failed than data.max_failed_fraction allows."""

# ---- 1. patients -------------------------------------------------------------------------


def build_patients(cfg, demographics_path):
    """One row per participant: which wrist to use, DBS, and the bias slices. Saved in interim/."""
    demo = read_demographics(demographics_path, cfg)
    side = demo.get("pd_dominantside", pd.Series(np.nan, index=demo.index)).astype("string").str.lower()
    choose, fallback = cfg["wrist"]["choose"], cfg["wrist"]["fallback"]
    wrist = side if choose == "more_affected" else pd.Series(choose, index=demo.index)
    wrist = wrist.where(wrist.isin(["left", "right"]), fallback)  # "both" or blank
    slices = [c for c in cfg["bias"]["slice_columns"] if c in demo.columns and c != "dbs"]
    patients = pd.DataFrame({
        "participant_id": demo["participant_id"],
        "wrist": wrist,
        "more_affected_side": ((wrist == side) | (side == "both")).fillna(False).astype(bool),
        "dbs": demo["dbs"].astype(str).str.lower().map(BOOL_TEXT) if "dbs" in demo else np.nan,
    }).join(demo[slices])
    return write_table(patients, cfg["resolved_paths"]["interim_dir"] / "patients.csv")


def patients_step(cfg, manifest):
    """Get Demographics.csv and build the patients table; returns its path."""
    entry = manifest.get("demographics")
    if not entry:
        raise acquire.AcquisitionError("No Demographics.csv: each participant's wrist and DBS come from it")
    return str(build_patients(cfg, acquire.download_file(entry, cfg)))

# ---- 2. one participant ------------------------------------------------------------------


def _read_hour(archive, zip_name, csv_name, hour, sep):
    """One hour of one wrist, trimmed to the diary hour, and a status."""
    if pd.isna(zip_name) and pd.isna(csv_name):
        return None, "no_file"
    try:
        df = archive.accelerometer(zip_name, csv_name, sep)
    except FileNotFoundError:
        return None, "missing_file"
    except (zipfile.BadZipFile, ValueError, pd.errors.ParserError) as exc:
        logger.warning("Corrupt accelerometer file %s: %s", zip_name, exc)
        return None, "corrupt_file"
    start = hour.hour_start * SECONDS_PER_HOUR
    return df[(df["Time"] >= start) & (df["Time"] < start + SECONDS_PER_HOUR)], "ok"


def drop_reasons(hours, cfg):
    """Why the hourly baselines should skip an hour ("" = use it). The first rule that applies wins."""
    c = cfg["hours"]
    rules = [
        ("no_label", hours["kinesia_score"].isna() & hours["kinesia_text"].ne(SLEEP)),
        ("not_worn", hours["wear_status"].eq(NOT_WORN)),
        ("no_sensor_data", hours["W_status"].ne("ok")),
        ("low_valid_fraction", hours["W_valid_fraction"] < c["min_valid_fraction"]),
        ("sensor_nonwear", hours["worn_fraction"].fillna(0) < c["min_worn_fraction"]),
    ]
    reason = pd.Series("", index=hours.index, dtype=object)
    for name, mask in rules:
        reason[mask & reason.eq("")] = name
    return reason


def process_participant(zip_path, cfg, patients):
    """Diary, doses, windows and hourly features for one participant, written to interim/<pid>/."""
    pid = participant_id(zip_path)
    row = patients.loc[patients["participant_id"] == pid]
    if row.empty or row["wrist"].isna().any():
        raise ValueError(f"{pid}: no wrist in the demographics (pd_dominantside)")
    sep, n_agree = cfg["files"]["csv_separator"], cfg["checks"]["agreement_hours"]

    with Archive(zip_path) as archive:
        hours, label_issues = parse_diary(archive.diary(sep), cfg)
        events, dose_issues = dose_events(hours, LeddTable.load(cfg), cfg)
        # Some participants have only one wrist recorded: use it, even if it's the less affected one.
        wrist, more_affected = row["wrist"].iloc[0], bool(row["more_affected_side"].iloc[0])
        other = "right" if wrist == "left" else "left"
        recorded = {side: any(archive.has(z) or archive.has(c)
                              for z, c in zip(hours[f"{side}_zip"], hours[f"{side}_csv"]))
                    for side in (wrist, other)}
        if not recorded[wrist] and recorded[other]:
            logger.warning("%s: no %s-wrist recording; using the %s wrist", pid, wrist, other)
            wrist, other, more_affected = other, wrist, not more_affected
        feats, parts, other_min = [], [], []
        for h in hours.itertuples(index=False):
            df, status = _read_hour(archive, getattr(h, f"{wrist}_zip"), getattr(h, f"{wrist}_csv"), h, sep)
            if df is None:
                feats.append(sensor.empty_features(status))
                continue
            offset = h.day * SECONDS_PER_DAY
            feats.append(sensor.hour_features(df, cfg))
            win = sensor.hour_windows(df, offset, cfg)
            win["diary_worn"] = np.full(len(win["time_s"]), h.wear_status != NOT_WORN)
            parts.append(win)
            # A few waking, worn hours of the other wrist, to confirm the orientation fix.
            if len(other_min) < n_agree and pd.notna(h.kinesia_score) and h.wear_status != NOT_WORN:
                odf, _ = _read_hour(archive, getattr(h, f"{other}_zip"), getattr(h, f"{other}_csv"), h, sep)
                if odf is not None:
                    other_min.append(sensor.minute_means(odf, offset))

    win = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else \
        {**sensor.empty_windows(cfg), "diary_worn": np.zeros(0, bool)}
    worn = sensor.wear_flags(win["time_s"], win["signal"], win["temp"], win["diary_worn"], cfg)
    hour_of = win["time_s"] // SECONDS_PER_HOUR
    # Orientation, decided per wear bout: a watch can only be turned round while it's off.
    awake = np.isin(hour_of, hours.loc[hours["kinesia_score"].notna(), "hour_index"])
    bouts = sensor.wear_bouts(win["time_s"], worn, cfg["orientation"]["rewear_gap_minutes"] * 60)
    turned = sensor.turned_by_bout(win["mean"], worn, awake, bouts, cfg)
    mirrored = sensor.orient(win["signal"], turned, wrist, cfg)
    sensor.orient(win["mean"], turned, wrist, cfg)
    chosen_min = pd.DataFrame(win["mean"][worn], columns=sensor.AXES).groupby(win["time_s"][worn] // 60).mean()
    agreement = sensor.wrist_agreement(chosen_min, pd.concat(other_min), other, cfg) if other_min \
        else {"minutes": 0}
    turned_share = float(turned[worn].mean()) if worn.any() else 0.0

    per_hour = pd.DataFrame({"hour_index": hour_of, "worn": worn}).groupby("hour_index")["worn"] \
        .agg(n_windows="size", worn_fraction="mean").reset_index()
    hours = pd.concat([hours, pd.DataFrame(feats).add_prefix("W_")], axis=1) \
        .merge(per_hour, on="hour_index", how="left").fillna({"n_windows": 0}).drop(columns=FILE_COLUMNS)
    hours["drop_reason"] = drop_reasons(hours, cfg)

    out = cfg["resolved_paths"]["interim_dir"] / pid
    shutil.rmtree(out, ignore_errors=True)
    (out / "windows").mkdir(parents=True)
    write_table(hours, out / "hours.csv")
    write_table(events, out / "events.csv")
    for name, values in [("time_s", win["time_s"]), ("worn", worn), ("signal", win["signal"])]:
        np.save(out / "windows" / f"{name}.npy", values)
    write_json({"participant_id": pid, "wrist": wrist, "more_affected_side": more_affected,
                "turned_round_share": turned_share, "wear_bouts": int(len(np.unique(bouts[worn]))),
                "mirrored": mirrored, "agreement": agreement,
                "label_issues": label_issues, "dose_issues": dose_issues}, out / "meta.json")
    logger.info("%s (%s wrist, %.0f%% turned round, %d wear bouts): %d hours, %d windows (%d worn), %d doses",
                pid, wrist, 100 * turned_share, len(np.unique(bouts[worn])), len(hours), len(worn), worn.sum(),
                len(events))
    return {"participant_id": pid, "dir": str(out)}


def participant_step(cfg, entry, patients_path):
    """Download one participant, process it, then delete the zip."""
    zip_path = acquire.download_file(entry, cfg)
    result = process_participant(zip_path, cfg, pd.read_csv(patients_path))
    if not entry.get("local_path") and cfg["data"]["delete_raw_after_processing"]:
        Path(zip_path).unlink(missing_ok=True)
    return result


def _attempt(cfg, entry, patients_path):
    """(result, None), or (None, error text): a worker's failure reaches the parent as text."""
    try:
        return participant_step(cfg, entry, patients_path), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def process_all(cfg, participants, patients_path):
    """Every participant, run.parallel_participants at a time. Failures are collected, not raised."""
    workers = min(cfg["run"]["parallel_participants"], len(participants))
    args = [[cfg] * len(participants), participants, [patients_path] * len(participants)]
    if workers <= 1:
        outcomes = list(map(_attempt, *args))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(_attempt, *args))
    results, failures = [], []
    for entry, (result, error) in zip(participants, outcomes):
        if error is None:
            results.append(result)
        else:
            logger.error("Skipping %s: %s", entry["name"], error)
            failures.append({"participant_id": Path(entry["name"]).stem, "error": error})
    return results, failures


def record_failures(cfg, failures, total):
    """Save and alert on the failed participants. True when more failed than allowed."""
    write_json(failures, cfg["resolved_paths"]["reports_dir"] / "failed_participants.json")
    if not failures:
        return False
    notify(f"[COPS pipeline] {len(failures)} of {total} participants failed",
           "\n".join(f"{f['participant_id']}: {f['error']}" for f in failures), cfg)
    return len(failures) / total > cfg["data"]["max_failed_fraction"]


def check_failures(cfg, manifest, results):
    """For the DAG: once every participant has finished or failed, keep the ones that worked."""
    results = [r for r in results if r]
    done = {r["participant_id"] for r in results}
    failures = [{"participant_id": Path(p["name"]).stem, "error": "failed after its retries; see its task log"}
                for p in manifest["participants"] if Path(p["name"]).stem not in done]
    if record_failures(cfg, failures, len(manifest["participants"])) or not results:
        raise TooManyFailures(f"{len(failures)} of {len(manifest['participants'])} participants failed")
    return results

# ---- 3. the dataset ----------------------------------------------------------------------


def build(cfg, results, patients_path, mode=None):
    """Join every participant into the outputs in data/processed/."""
    processed, reports = cfg["resolved_paths"]["processed_dir"], cfg["resolved_paths"]["reports_dir"]
    dirs = [Path(r["dir"]) for r in sorted(results, key=lambda r: r["participant_id"])]
    metas = [read_json(d / "meta.json") for d in dirs]
    patients = pd.read_csv(patients_path)

    hours = pd.concat([read_table(d / "hours.csv") for d in dirs], ignore_index=True)
    slices = [c for c in cfg["bias"]["slice_columns"] if c in patients.columns and c not in hours.columns]
    hours = hours.merge(patients[["participant_id"] + slices], on="participant_id", how="left")
    events = pd.concat([read_table(d / "events.csv") for d in dirs], ignore_index=True)
    # The wrist actually used (it falls back when only one was recorded) and its orientation fixes.
    fixes = pd.DataFrame([{k: m[k] for k in FIX_COLUMNS} for m in metas])
    patients = patients.drop(columns=["wrist", "more_affected_side"]).merge(fixes, on="participant_id")[PATIENT_COLUMNS]

    shutil.rmtree(processed / "windows", ignore_errors=True)
    for d, m in zip(dirs, metas):
        shutil.move(str(d / "windows"), str(processed / "windows" / m["participant_id"]))
    worn = [np.load(processed / "windows" / m["participant_id"] / "worn.npy") for m in metas]
    # How the windows were cut, next to them: Model_Pipeline's loader refuses windows without it.
    write_json({k: cfg["windows"][k] for k in ("spec_version", "seconds", "rate_hz", "axes")},
               processed / "windows" / "manifest.json")

    write_table(hours, processed / "hours.csv")
    write_table(events, processed / "events.csv")
    write_table(patients, processed / "patients.csv")
    write_json([{**i, "participant_id": m["participant_id"]} for m in metas
                for i in m["label_issues"] + m["dose_issues"]], reports / "label_issues.json")
    write_table(pd.DataFrame([{"participant_id": m["participant_id"], "wrist": m["wrist"],
                               "turned_round_share": m["turned_round_share"], "wear_bouts": m["wear_bouts"],
                               **m["agreement"]} for m in metas]), reports / "wrist_agreement.csv")
    usable = hours["drop_reason"].isna()
    write_json({**cfg["windows"], "wrist": cfg["wrist"]["choose"], "mode": mode,
                "participants": len(metas), "hours": len(hours), "usable_hours": int(usable.sum()),
                "windows": int(sum(len(w) for w in worn)), "worn_windows": int(sum(w.sum() for w in worn)),
                "doses": len(events), "created_at": datetime.now(timezone.utc).isoformat()},
               processed / "dataset_meta.json")
    logger.info("Dataset: %d participants, %d hours (%d usable), %d windows", len(metas), len(hours),
                usable.sum(), sum(len(w) for w in worn))
    return processed

# ---- command line ------------------------------------------------------------------------


def run(cfg, mode=None):
    """The whole pipeline. Returns 0 on success, 1 when it stopped."""
    manifest = acquire.fetch_manifest(cfg, mode)
    patients_path = patients_step(cfg, manifest)
    results, failures = process_all(cfg, manifest["participants"], patients_path)
    if record_failures(cfg, failures, len(manifest["participants"])) or not results:
        logger.error("Stopping: too many participants failed (see failed_participants.json)")
        return 1
    build(cfg, results, patients_path, manifest["mode"])
    try:
        checks.run_checks(cfg)
    except checks.DataCheckError as exc:
        logger.error("Stopping: the dataset failed its checks:\n%s", exc)
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="COPS data pipeline")
    parser.add_argument("--mode", choices=["sample", "full"], default=None)
    parser.add_argument("--config", default=None, help="path to a config YAML")
    args = parser.parse_args(argv)
    return run(load_config(args.config), args.mode)


if __name__ == "__main__":
    sys.exit(main())
