"""How much patients differ from each other: a per-patient profile and a between/within summary.

    python -m scripts.reporting.patient_report

Writes to <artifacts_dir>/reports/patients/ (not in Git: it names patients):
    patient_profile.csv   one row per patient: W_ means and stds, label shares, doses, peak hours
    feature_spread.csv    per W_ feature: spread of patient means vs spread within a patient
    patient_report.md     the two tables, readable
"""

import numpy as np
import pandas as pd

from scripts.data.baseline_loader import FEATURE_NAMES, build_features, good_sensor, settings
from scripts.utils.config import load_config

SLEEP = "Sleep"


def feature_settings(cfg):
    """The baseline candidate's feature settings, defaults filled in."""
    return settings(((cfg.get("baseline") or {}).get("candidate") or {}).get("params", {}).get("features"))


def _peak_hour(hour_start, mask):
    """The hour of day when mask is most often true (NaN if never)."""
    counts = hour_start[mask].value_counts()
    return int(counts.idxmax()) if len(counts) else np.nan


def patient_profile(hours, events, table, cfg):
    good = pd.Series(good_sensor(hours, feature_settings(cfg)), index=hours.index)
    awake = hours["kinesia_score"].notna()
    doses = events[events["kind"] == "dose"]
    rows = []
    for pid, h in hours.groupby("participant_id"):
        g, a = good.loc[h.index], awake.loc[h.index]
        score, tremor = h["kinesia_score"], h["tremor_score"]
        d = doses[doses["participant_id"] == pid]
        days = h["day"].nunique()
        t = table[table["participant_id"] == pid].set_index("hour_index").reindex(h["hour_index"]).set_axis(h.index)
        row = {"participant_id": pid, "hours": len(h), "good_sensor_hours": int(g.sum()),
               "awake_labelled_hours": int(a.sum()), "sleep_hours": int(h["kinesia_text"].eq(SLEEP).sum())}
        for name in FEATURE_NAMES:
            x = h.loc[g, f"W_{name}"]
            row[f"{name}_mean"], row[f"{name}_std"] = x.mean(), x.std()
        row.update({
            "off_share": (score[a] < 0).mean(), "on_share": (score[a] == 0).mean(),
            "dyskinetic_share": (score[a] > 0).mean(), "tremor_share": (tremor[a] > 0).mean(),
            "peak_off_hour": _peak_hour(h["hour_start"], a & (score < 0)),
            "peak_tremor_hour": _peak_hour(h["hour_start"], a & (tremor > 0)),
            "doses_per_day": len(d) / days if days else np.nan,
            "ledd_per_day": d["ledd"].sum() / days if days else np.nan,
            "mean_dose_ledd": d["ledd"].mean(),
            "dose_hours": " ".join(str(x) for x in sorted(((d["time_s"] // 3600) % 24).astype(int).unique())),
            "hours_since_dose_off": t.loc[a & (score < 0), "hours_since_dose"].median(),
            "hours_since_dose_on": t.loc[a & (score == 0), "hours_since_dose"].median(),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def feature_spread(hours, cfg):
    """Per feature: std of the patients' means vs the average std within a patient.

    A ratio above 1 means patients differ from each other more than one patient's hours
    differ from each other, so per-patient z-scores should help.
    """
    good = hours[good_sensor(hours, feature_settings(cfg))]
    rows = []
    for name in FEATURE_NAMES:
        by = good.groupby("participant_id")[f"W_{name}"]
        between, within = by.mean().std(), by.std().mean()
        rows.append({"feature": name, "overall_mean": good[f"W_{name}"].mean(), "between_patient_std": between,
                     "within_patient_std": within, "ratio": between / within if within else np.nan})
    return pd.DataFrame(rows)


def _markdown(table):
    """A Markdown table (pandas' to_markdown needs tabulate, which the pipeline doesn't install)."""
    lines = ["| " + " | ".join(map(str, table.columns)) + " |", "|" + "---|" * len(table.columns)]
    for row in table.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def write_report(cfg):
    paths = cfg["resolved_paths"]
    hours = pd.read_csv(paths["hours_csv"], low_memory=False)
    events = pd.read_csv(paths["events_csv"])
    table = build_features(hours, events, feature_settings(cfg))
    profile, spread = patient_profile(hours, events, table, cfg), feature_spread(hours, cfg)
    out = paths["artifacts_dir"] / "reports" / "patients"
    out.mkdir(parents=True, exist_ok=True)
    profile.to_csv(out / "patient_profile.csv", index=False)
    spread.to_csv(out / "feature_spread.csv", index=False)
    short = ["participant_id", "awake_labelled_hours", "move_mean_mean", "tremor_mean_mean", "off_share",
             "dyskinetic_share", "tremor_share", "peak_off_hour", "peak_tremor_hour", "doses_per_day",
             "ledd_per_day", "hours_since_dose_off", "hours_since_dose_on"]
    (out / "patient_report.md").write_text(
        f"# Patient profiles\n\n{len(profile)} patients.\n\n## Feature spread\n\n"
        "ratio > 1: patients differ more from each other than within themselves.\n\n"
        f"{_markdown(spread.round(4))}\n\n## Per patient\n\n"
        f"{_markdown(profile[short].round(3))}\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    print(f"Written to {write_report(load_config())}")
