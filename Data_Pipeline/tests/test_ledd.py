import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.ledd import LeddError, LeddTable, dose_events, normalise_name
from scripts.utils import PROJECT_ROOT
from tests.helpers import make_config

TABLE = PROJECT_ROOT / "config" / "ledd.yaml"
RAW = yaml.safe_load(TABLE.read_text(encoding="utf-8"))


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path, medication={"ledd_table": str(TABLE)})


@pytest.fixture
def table(cfg):
    return LeddTable.load(cfg)


def table_with(**cops_names):
    """The real table, with extra COPS names, for drugs COPS doesn't use (yet)."""
    raw = copy.deepcopy(RAW)
    raw["names"]["cops"].update(cops_names)
    return LeddTable.from_dict(raw)


def diary(*hours):
    """Rows of (participant, day, hour_start, medication, dosage_mg)."""
    return pd.DataFrame(hours, columns=["participant_id", "day", "hour_start", "medication", "dosage_mg"])


def ledd_of(events, drug):
    return events.loc[events["drug"] == drug, "ledd"].tolist()


def test_names_match_however_cops_spaces_them():
    assert normalise_name("Levodopa / Benserazid  RE") == normalise_name("levodopa/benserazid re")


def test_every_drug_is_cited_to_a_listed_reference():
    references = set(RAW["references"])
    for drug_id, d in RAW["drugs"].items():
        assert d.get("cite", "").split()[0] in references, drug_id


@pytest.mark.parametrize("name, mg, expected", [
    ("Levodopa / Benserazid", 100, 100),               # Table 1: Levodopa, DD x 1
    ("Levodopa / Benserazid RE", 100, 75),             # controlled release, DD x 0.75
    ("Levodopa / Carbidopa / Entacapone", 75, 99.75),  # levodopa plus entacapone's LD x 0.33
    ("Ropinirole ER", 2, 40),                          # DD x 20
    ("Pramipexole RE", 2.1, 300.006),                  # base, DD x 142.86
    ("Safinamide", 50, 150),                           # LED = 150 for 50 and 100 mg/d
    ("Safinamide", 100, 150),
])
def test_each_rule(table, cfg, name, mg, expected):
    events, _ = dose_events(diary(("P1", 0, 8, name, str(mg))), table, cfg)
    assert events["ledd"].tolist() == pytest.approx([expected])


def test_doses_go_at_mid_hour_on_the_diary_clock(table, cfg):
    events, _ = dose_events(diary(("P1", 1, 8, "Levodopa / Carbidopa", "100")), table, cfg)
    assert events["time_s"].tolist() == [86400 + 8 * 3600 + 30 * 60]
    assert events[["kind", "source"]].iloc[0].tolist() == ["dose", "on_time"]


def test_several_drugs_in_one_hour_are_one_event_each(table, cfg):
    events, _ = dose_events(diary(("P1", 0, 8, "Amantadine;Levodopa / Carbidopa", "100;200")), table, cfg)
    assert sorted(events["drug"]) == ["amantadine", "levodopa_ir"]


def test_a_fixed_amount_counts_once_a_day(table, cfg):
    events, _ = dose_events(diary(
        ("P1", 0, 8, "Safinamide", "50"), ("P1", 0, 20, "Safinamide", "50"), ("P1", 1, 8, "Safinamide", "50"),
    ), table, cfg)
    assert ledd_of(events, "safinamide") == [150, 150]   # day 0 once, day 1 once


def test_opicapone_boosts_every_levodopa_dose_that_day_and_adds_no_event(table, cfg):
    events, _ = dose_events(diary(
        ("P1", 0, 8, "Levodopa / Carbidopa;Amantadine", "100;100"),   # same day, before it: boosted
        ("P1", 0, 20, "Opicapone", "50"),
        ("P1", 1, 8, "Levodopa / Carbidopa", "100"),                  # next day, no opicapone: not
        ("P2", 0, 8, "Levodopa / Carbidopa", "100"),                  # another participant: not
    ), table, cfg)
    assert "opicapone" not in set(events["drug"])
    p1 = events[events["participant_id"] == "P1"]
    assert ledd_of(p1, "levodopa_ir") == pytest.approx([150, 100])
    assert ledd_of(p1, "amantadine") == [100]                         # only levodopa is boosted
    assert ledd_of(events[events["participant_id"] == "P2"], "levodopa_ir") == [100]


@pytest.mark.parametrize("booster, hours, factor", [("entacapone", 4, 0.33), ("tolcapone", 8, 0.5)])
def test_entacapone_and_tolcapone_boost_only_the_hours_after_them(cfg, booster, hours, factor):
    events, _ = dose_events(diary(
        ("P1", 0, 7, "Levodopa / Carbidopa", "100"),                  # before it: not
        ("P1", 0, 8, f"{booster};Levodopa / Carbidopa", "100;100"),   # with it: boosted
        ("P1", 0, 8 + hours - 1, "Levodopa / Carbidopa", "100"),      # inside the window: boosted
        ("P1", 0, 8 + hours, "Levodopa / Carbidopa", "100"),          # window over: not
    ), table_with(**{booster: booster}), cfg)
    boosted = 100 * (1 + factor)
    assert ledd_of(events, "levodopa_ir") == pytest.approx([100, boosted, boosted, 100])


def test_istradefylline_applies_on_top_of_the_comt_boost(cfg):
    events, _ = dose_events(diary(
        ("P1", 0, 8, "Opicapone;Istradefylline;Levodopa / Carbidopa", "50;20;100"),
    ), table_with(istradefylline="istradefylline"), cfg)
    assert ledd_of(events, "levodopa_ir") == pytest.approx([100 * 1.5 * 1.2])   # (LD + COMT) x 0.2 added


def test_a_booster_counts_once_even_if_taken_twice(table, cfg):
    events, _ = dose_events(diary(
        ("P1", 0, 7, "Opicapone", "50"), ("P1", 0, 8, "Opicapone;Levodopa / Carbidopa", "50;100"),
    ), table, cfg)
    assert ledd_of(events, "levodopa_ir") == pytest.approx([150])


def test_anticholinergics_add_nothing(cfg):
    events, _ = dose_events(diary(("P1", 0, 8, "Trihexyphenidyl;Amantadine", "2;100")),
                            table_with(trihexyphenidyl="trihexyphenidyl"), cfg)
    assert events["drug"].tolist() == ["amantadine"]


def test_an_unknown_drug_stops_the_run_and_names_it(table, cfg):
    with pytest.raises(LeddError, match="mystery pill"):
        dose_events(diary(("P1", 0, 8, "Mystery Pill", "10")), table, cfg)


def test_hours_it_cant_read_are_reported_not_guessed(table, cfg):
    events, issues = dose_events(diary(
        ("P1", 0, 8, "Amantadine;Levodopa / Carbidopa", "100"),   # 2 drugs, 1 dose
        ("P1", 0, 9, "Amantadine", "lots"),
        ("P1", 0, 10, "Amantadine", "100"),
        ("P1", 0, 11, np.nan, np.nan),                           # no medication: nothing to do
    ), table, cfg)
    assert [i["issue"] for i in issues] == ["dose_count_mismatch", "unreadable_dose"]
    assert len(events) == 1


CITE = {"cite": "jost2023 Table 1: X"}


@pytest.mark.parametrize("drug, match", [
    ({"rule": "guess", "factor": 1, **CITE}, "rule"),
    ({"rule": "dose", "factor": 1}, "needs a cite"),
    ({"rule": "dose", "factor": 1, "cite": "someone2024 Table 2"}, "needs a cite"),
    ({"rule": "dose", **CITE}, "positive factor"),
    ({"rule": "dose", "factor": -1, **CITE}, "positive factor"),
    ({"rule": "levodopa_boost", "factor": 0.5, **CITE}, "boost"),
    ({"rule": "levodopa_boost", "factor": 0.5, "boost": 4, "of": "dopamine", **CITE}, "boosts"),
])
def test_an_incomplete_or_uncited_row_is_refused(drug, match):
    with pytest.raises(LeddError, match=match):
        LeddTable.from_dict({"references": RAW["references"], "drugs": {"x": drug}})


def test_a_name_for_an_unknown_drug_is_refused():
    with pytest.raises(LeddError, match="unknown drug"):
        LeddTable.from_dict({"references": RAW["references"], "drugs": {}, "names": {"cops": {"pill": "y"}}})


def test_on_real_table_if_present(table, cfg):
    path = Path(__file__).resolve().parents[1] / "data/processed/hours.csv"
    if not path.exists():
        pytest.skip("data pipeline output not pulled")
    hours = pd.read_csv(path, low_memory=False)
    events, issues = dose_events(hours, table, cfg)
    assert not issues
    assert (events["ledd"] > 0).all()
    assert set(events["participant_id"]) <= set(hours.loc[hours["medication"].notna(), "participant_id"])
