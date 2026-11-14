"""Checks on the built dataset, and its reports.

    critical   the contract Model_Pipeline's core/data.py relies on. Any problem
               stops the run (and the DAG) and sends an alert.
    warnings   worth a look but not wrong: thin participants, mostly unworn
               recordings, wrists that still disagree after the orientation fix,
               diary label issues. Sent as an alert; the run carries on.
    bias       label balance per group (sex, age group, ...), for the hours the
               baselines use. The model pipeline checks the model itself per group.

Writes reports/checks.json, reports/bias_report.json and reports/bias_report.md.
"""

import numpy as np
import pandas as pd

from scripts.diary import SLEEP
from scripts.utils import get_logger, notify, read_json, read_table, write_json

logger = get_logger("checks")

KINESIA_SCORES, TREMOR_SCORES = tuple(range(-3, 4)), (0, 1, 2)
DOSE_SOURCES = ("on_time", "late", "none_logged")
REQUIRED = {
    "hours.csv": ["participant_id", "hour_index", "kinesia_text", "kinesia_score", "tremor_score", "drop_reason"],
    "events.csv": ["participant_id", "time_s", "kind", "ledd", "source", "kinesia_score", "tremor_score"],
    "patients.csv": ["participant_id", "more_affected_side", "dbs"],
}
STATES = {"off": (-3, -2, -1), "on": (0,), "dyskinetic": (1, 2, 3)}


class DataCheckError(RuntimeError):
    """The dataset breaks the contract Model_Pipeline relies on."""


def _bad(values, allowed):
    return int((~values.dropna().isin(allowed)).sum())


def table_problems(hours, events, patients):
    """Problems with the three tables, as sentences."""
    p = []
    for name, table in [("hours.csv", hours), ("events.csv", events), ("patients.csv", patients)]:
        missing = [c for c in REQUIRED[name] if c not in table.columns]
        if missing:
            return [f"{name} is missing columns {missing}"]
    if hours.empty:
        p.append("hours.csv has no rows")
    if hours.duplicated(["participant_id", "hour_index"]).any():
        p.append("hours.csv has a participant-hour twice")
    if _bad(hours["kinesia_score"], KINESIA_SCORES):
        p.append("hours.csv has a kinesia_score outside -3..3")
    if _bad(hours["tremor_score"], TREMOR_SCORES):
        p.append("hours.csv has a tremor_score outside 0..2")

    if not events["kind"].eq("dose").all():
        p.append("events.csv has an event that isn't a dose")
    if not np.isfinite(pd.to_numeric(events["time_s"], errors="coerce")).all():
        p.append("events.csv has a dose without a time")
    ledd = pd.to_numeric(events["ledd"], errors="coerce")
    if not (np.isfinite(ledd) & (ledd >= 0)).all():
        p.append("events.csv has a dose without an LEDD of 0 or more")
    if _bad(events["source"], DOSE_SOURCES) or events["source"].isna().any():
        p.append(f"events.csv has a source outside {DOSE_SOURCES}")
    if not set(events["participant_id"]) <= set(hours["participant_id"]):
        p.append("events.csv has a participant who isn't in hours.csv")

    if patients["participant_id"].duplicated().any():
        p.append("patients.csv has a participant twice")
    missing = set(hours["participant_id"]) - set(patients["participant_id"])
    if missing:
        p.append(f"patients.csv has no row for {sorted(missing)}")
    for col in ("more_affected_side", "dbs"):
        bad = patients.loc[~patients[col].isin([True, False]), "participant_id"]
        if len(bad):
            p.append(f"patients.csv {col} isn't true or false for {sorted(bad)}")
    return p


def window_problems(folder, pid, hour_index, cfg):
    """Problems with one participant's windows, and their counts."""
    w = cfg["windows"]
    try:
        time_s, worn, signal = (np.load(folder / f"{n}.npy", mmap_mode="r") for n in ("time_s", "worn", "signal"))
    except FileNotFoundError:
        return [f"{pid}: windows are missing"], {}
    n, p = len(time_s), []
    if signal.shape != (n, w["axes"], w["seconds"] * w["rate_hz"]) or signal.dtype != np.float32:
        p.append(f"{pid}: signal is {signal.shape} {signal.dtype}, not ({n}, {w['axes']}, "
                 f"{w['seconds'] * w['rate_hz']}) float32")
    if worn.shape != (n,) or worn.dtype != bool:
        p.append(f"{pid}: worn isn't {n} booleans")
    if n and not ((np.diff(time_s) >= w["seconds"]).all() and (np.asarray(time_s) % w["seconds"] == 0).all()):
        p.append(f"{pid}: time_s doesn't step by whole windows")
    if n and not np.isfinite(signal).all():
        p.append(f"{pid}: signal has NaN or infinite values")
    if not set(np.unique(np.asarray(time_s) // 3600)) <= set(hour_index):
        p.append(f"{pid}: windows fall outside the diary hours")
    return p, {"windows": n, "worn_fraction": float(np.mean(worn)) if n else 0.0}


def state(hours):
    """The broad state of each hour: off, on, dyskinetic, asleep, or None."""
    out = pd.Series(None, index=hours.index, dtype=object)
    for name, scores in STATES.items():
        out[hours["kinesia_score"].isin(scores)] = name
    out[hours["kinesia_text"].eq(SLEEP)] = "asleep"
    return out


def find_warnings(hours, stats, agreement, issues, cfg):
    usable = hours[hours["drop_reason"].isna()]
    counts = usable.groupby("participant_id").size().reindex(hours["participant_id"].unique(), fill_value=0)
    minimum = cfg["checks"]["min_labelled_hours"]
    out = [f"{pid}: only {n} usable hours (minimum {minimum})" for pid, n in counts.items() if n < minimum]
    out += [f"{pid}: only {s['worn_fraction']:.0%} of windows worn" for pid, s in stats.items()
            if s and s["worn_fraction"] < cfg["hours"]["min_worn_fraction"]]
    for a in agreement.itertuples():
        corrs = [getattr(a, f"corr_{x}", np.nan) for x in "xyz"]
        if not a.minutes:
            out.append(f"{a.participant_id}: wrist agreement not checked (no hours with both wrists)")
        elif min((c for c in corrs if not np.isnan(c)), default=0) < 0:
            out.append(f"{a.participant_id}: wrists still disagree after the orientation fix "
                       f"(x {corrs[0]:.2f}, y {corrs[1]:.2f}, z {corrs[2]:.2f})")
    if issues:
        for (field, issue), n in pd.DataFrame(issues).fillna({"field": "medication"}) \
                .groupby(["field", "issue"]).size().items():
            out.append(f"diary {field}: {n} x {issue}")
    return out


def bias_report(hours, cfg):
    """Rows, participants and label shares per group, for the usable labelled hours."""
    usable = hours[hours["drop_reason"].isna()].assign(state=lambda h: state(h))
    report = {}
    for col in [c for c in cfg["bias"]["slice_columns"] if c in usable.columns]:
        groups = usable[col].astype(str).replace("nan", "missing")
        shares = pd.crosstab(groups, usable["state"], normalize="index").round(3)
        sizes = usable.groupby(groups).agg(rows=("participant_id", "size"), participants=("participant_id", "nunique"))
        report[col] = sizes.join(shares).reset_index(names=col).to_dict("records")
    return report


def bias_markdown(report):
    out = ["# Bias report: label balance per group\n",
           "Usable labelled hours (drop_reason empty). Shares of off, on, dyskinetic and asleep per group.\n"]
    for col, records in report.items():
        cols = list(records[0]) if records else []
        out += [f"## {col}\n", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        out += ["| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |" for r in records]
        out.append("")
    return "\n".join(out)


def run_checks(cfg):
    """Check the built dataset, write the reports, alert, and raise DataCheckError on any critical problem."""
    processed, reports = cfg["resolved_paths"]["processed_dir"], cfg["resolved_paths"]["reports_dir"]
    hours, events, patients = (read_table(processed / n) for n in ("hours.csv", "events.csv", "patients.csv"))
    critical = table_problems(hours, events, patients)

    meta = read_json(processed / "dataset_meta.json")
    spec = {k: cfg["windows"][k] for k in ("spec_version", "seconds", "rate_hz", "axes")}
    if {k: meta.get(k) for k in spec} != spec:
        critical.append(f"windows were cut with spec {meta}, but the config says {spec}")
    stats = {}
    for pid, group in hours.groupby("participant_id"):
        problems, stats[pid] = window_problems(processed / "windows" / pid, pid, set(group["hour_index"]), cfg)
        critical += problems

    warnings = find_warnings(hours, stats, read_table(reports / "wrist_agreement.csv"),
                             read_json(reports / "label_issues.json"), cfg)
    bias = bias_report(hours, cfg)
    write_json({"critical": critical, "warnings": warnings, "windows": stats}, reports / "checks.json")
    write_json(bias, reports / "bias_report.json")
    (reports / "bias_report.md").write_text(bias_markdown(bias), encoding="utf-8")
    logger.info("Checks: %d critical, %d warnings", len(critical), len(warnings))

    if critical:
        notify("[COPS pipeline] Dataset failed its checks", "\n".join(critical), cfg)
        raise DataCheckError("\n".join(critical))
    if warnings:
        notify("[COPS pipeline] Dataset warnings", "\n".join(warnings[:30]), cfg)
    return {"critical": critical, "warnings": warnings}
