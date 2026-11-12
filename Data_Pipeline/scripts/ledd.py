"""Turn the diary's medication into dose events, each with its levodopa equivalent dose (LEDD).

The conversion table is config/ledd.yaml, from Jost et al. 2023 (Movement Disorders,
doi:10.1002/mds.29410), Table 1: every drug's rule and factor, where each value is cited,
and how each data source writes drug names. This module only applies it.

    table = LeddTable.load(cfg)
    events, issues = dose_events(diary, table, cfg)

One event per dose, at its time on the diary clock (seconds since the participant's first
diary day began). COPS only ticks the hour, so a dose goes at the hour's start plus
medication.dose_offset_minutes.

Levodopa boosters add no event of their own; they raise the levodopa doses they cover, as
Table 1's footnotes say: opicapone every levodopa dose that day, entacapone the doses in the
4 hours after it, tolcapone 8 hours. Istradefylline is applied last, on the levodopa dose
with its COMT boosts (footnote d).

Run it on the pipeline's table to see what the conversion gives:

    python -m scripts.ledd data/processed/hourly_all.csv
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import yaml

from scripts.utils.config import PROJECT_ROOT, load_config
from scripts.utils.logging_setup import get_logger

logger = get_logger("ledd")

RULES = ("dose", "fixed", "levodopa_boost", "none")
BOOST_OF = ("levodopa", "levodopa_and_comt")
WHOLE_DAY = "whole_day"
EVENT_COLUMNS = ["participant_id", "time_s", "kind", "ledd", "source", "kinesia_score", "tremor_score", "drug"]


class LeddError(ValueError):
    """The LEDD table is malformed, or a dose can't be converted with it."""


@dataclass(frozen=True)
class Drug:
    id: str
    rule: str
    factor: Optional[float]
    cite: str
    levodopa: bool = False
    boost: Union[str, float, None] = None   # WHOLE_DAY or hours, for levodopa boosters
    of: str = "levodopa"


def normalise_name(name):
    """Lower-case, single spaces, no spaces around "/": "Levodopa / Benserazid" -> "levodopa/benserazid"."""
    name = re.sub(r"\s+", " ", str(name).strip().lower())
    return re.sub(r"\s*/\s*", "/", name)


class LeddTable:
    """The drugs, their rules and the name maps from config/ledd.yaml, checked on load."""

    def __init__(self, drugs, names):
        self.drugs, self.names = drugs, names

    @classmethod
    def load(cls, cfg):
        """Read the table named by cfg["medication"]["ledd_table"] (relative to the project root)."""
        path = Path(cfg["medication"]["ledd_table"])
        if not path.is_absolute():
            path = Path(cfg.get("project_root", PROJECT_ROOT)) / path
        with open(path, encoding="utf-8") as f:
            table = cls.from_dict(yaml.safe_load(f))
        logger.info("LEDD table %s: %d drugs, %d data sources", path.name, len(table.drugs), len(table.names))
        return table

    @classmethod
    def from_dict(cls, raw):
        references = raw.get("references") or {}
        drugs = {drug_id: _read_drug(drug_id, d, references) for drug_id, d in raw["drugs"].items()}
        names = {}
        for source, mapping in (raw.get("names") or {}).items():
            names[source] = {}
            for name, drug_id in mapping.items():
                if drug_id not in drugs:
                    raise LeddError(f"LEDD table: {source} name {name!r} maps to unknown drug {drug_id!r}")
                names[source][normalise_name(name)] = drug_id
        if not raw.get("checked_against_paper", False):
            logger.warning("LEDD table not checked against its references; see config/ledd.yaml")
        return cls(drugs, names)

    def drug(self, name, source):
        """The Drug a data source's name means, or None if the name isn't in the table."""
        drug_id = self.names.get(source, {}).get(normalise_name(name))
        return self.drugs[drug_id] if drug_id else None


def _read_drug(drug_id, d, references):
    """One table row as a Drug, refusing anything that isn't complete and cited."""
    rule, factor, cite = d.get("rule"), d.get("factor"), d.get("cite")
    if rule not in RULES:
        raise LeddError(f"LEDD table: {drug_id} has rule {rule!r}; known: {RULES}")
    # Every value must be citable: a cite starting with a listed reference key.
    if not cite or str(cite).split()[0] not in references:
        raise LeddError(f"LEDD table: {drug_id} needs a cite starting with one of {sorted(references)}")
    if rule != "none" and not (isinstance(factor, (int, float)) and not isinstance(factor, bool) and factor > 0):
        raise LeddError(f"LEDD table: {drug_id} needs a positive factor, not {factor!r}")
    boost = d.get("boost")
    if rule == "levodopa_boost":
        if not (boost == WHOLE_DAY or (isinstance(boost, (int, float)) and boost > 0)):
            raise LeddError(f"LEDD table: {drug_id} needs boost: {WHOLE_DAY} or a number of hours, not {boost!r}")
        if d.get("of", "levodopa") not in BOOST_OF:
            raise LeddError(f"LEDD table: {drug_id} boosts {d.get('of')!r}; known: {BOOST_OF}")
    return Drug(drug_id, rule, factor, cite, bool(d.get("levodopa")), boost, d.get("of", "levodopa"))


def dose_events(diary, table, cfg):
    """One dose event per drug and dose in the diary, with its LEDD.

    Args:
        diary: Hourly rows with participant_id, day, hour_start, medication and dosage_mg
            (the diary's ";"-separated names and doses, in the same order).
        table: A LeddTable.
        cfg: Pipeline configuration (the medication section).

    Returns:
        (DataFrame, list[dict]): The events (EVENT_COLUMNS), and an issue for every hour
        whose medication couldn't be read.

    Raises:
        LeddError: A drug name isn't in the table. Fixed by adding it to config/ledd.yaml,
            never by guessing.
    """
    med = cfg["medication"]
    source, offset_s = med["name_source"], med["dose_offset_minutes"] * 60
    rows, boosts, issues, unknown = [], [], [], set()
    fixed_days = set()   # (participant, day, drug): a fixed amount counts once per day

    dosed = diary[diary["medication"].notna()]
    for _, hour in dosed.iterrows():
        pid, day = hour["participant_id"], hour["day"]
        where = {"participant_id": pid, "day": day, "hour_start": hour["hour_start"]}
        names = [n for n in str(hour["medication"]).split(";") if n.strip()]
        doses = [d for d in str(hour["dosage_mg"]).split(";") if d.strip()] if pd.notna(hour["dosage_mg"]) else []
        if len(names) != len(doses):
            issues.append({**where, "issue": "dose_count_mismatch", "drugs": len(names), "doses": len(doses)})
            continue
        time_s = day * 86400 + hour["hour_start"] * 3600 + offset_s
        for name, dose in zip(names, doses):
            drug = table.drug(name, source)
            if drug is None:
                unknown.add(normalise_name(name))
                continue
            mg = pd.to_numeric(dose, errors="coerce")
            if pd.isna(mg) or mg < 0:
                issues.append({**where, "issue": "unreadable_dose", "drug": drug.id})
            elif drug.rule == "levodopa_boost":
                boosts.append((pid, day, time_s, drug))
            elif drug.rule == "dose":
                rows.append({"participant_id": pid, "day": day, "time_s": time_s, "drug": drug,
                             "base": mg * drug.factor})
            elif drug.rule == "fixed" and (pid, day, drug.id) not in fixed_days:
                fixed_days.add((pid, day, drug.id))
                rows.append({"participant_id": pid, "day": day, "time_s": time_s, "drug": drug, "base": drug.factor})

    if unknown:
        raise LeddError(f"drug name(s) not in the LEDD table for {source}: {sorted(unknown)}; "
                        "add them to config/ledd.yaml")
    events = _apply_boosts(pd.DataFrame(rows, columns=["participant_id", "day", "time_s", "drug", "base"]), boosts)
    logger.info("Dose events: %d from %d dosed hours, %d boosted, %d hour(s) unreadable",
                len(events), len(dosed), int(events["boosted"].sum()), len(issues))
    return events.drop(columns="boosted"), issues


def _apply_boosts(events, boosts):
    """Raise the levodopa doses each booster covers (Jost 2023, Table 1 footnotes a and d).

    COMT inhibitors add factor x the levodopa LEDD. Istradefylline then adds factor x the
    levodopa LEDD with those COMT boosts. A booster counts once per dose, however many of
    its intakes cover it.
    """
    base = events["base"].astype(float).to_numpy()
    is_levodopa = events["drug"].map(lambda d: d.levodopa).astype(bool).to_numpy()
    comt = np.zeros(len(events))
    extra = np.zeros(len(events))
    boosted = np.zeros(len(events), dtype=bool)
    for of in BOOST_OF:   # COMT boosts first: istradefylline applies on top of them
        for booster in {b[3] for b in boosts if b[3].of == of}:
            covered = np.zeros(len(events), dtype=bool)
            for pid, day, start, drug in boosts:
                if drug is booster:
                    covered |= _covered(events, pid, day, start, drug) & is_levodopa
            amount = (base if of == "levodopa" else base + comt) * booster.factor
            (comt if of == "levodopa" else extra)[covered] += amount[covered]
            boosted |= covered
    events = events.assign(ledd=base + comt + extra, boosted=boosted, kind="dose", source="on_time",
                           kinesia_score=np.nan, tremor_score=np.nan, drug=events["drug"].map(lambda d: d.id))
    return events.sort_values(["participant_id", "time_s"]).reset_index(drop=True)[EVENT_COLUMNS + ["boosted"]]


def _covered(events, pid, day, start, drug):
    """The doses one booster intake covers: that whole day, or the hours after it."""
    same = (events["participant_id"] == pid).to_numpy()
    if drug.boost == WHOLE_DAY:
        return same & (events["day"] == day).to_numpy()
    t = events["time_s"].to_numpy()
    return same & (t >= start) & (t < start + drug.boost * 3600)


def main():
    parser = argparse.ArgumentParser(description="Convert a diary table's medication to dose events with LEDD.")
    parser.add_argument("table", help="a CSV with participant_id, day, hour_start, medication, dosage_mg")
    parser.add_argument("--out", help="write the events here as CSV")
    args = parser.parse_args()
    cfg = load_config()
    events, issues = dose_events(pd.read_csv(args.table, low_memory=False), LeddTable.load(cfg), cfg)
    print(events.groupby("drug")["ledd"].describe()[["count", "mean", "min", "max"]].to_string())
    print(f"{len(events)} events, {len(issues)} unreadable hour(s)")
    if args.out:
        events.to_csv(args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
