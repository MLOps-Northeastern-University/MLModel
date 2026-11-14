import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from scripts.core.errors import ConfigError, SchemaError
from scripts.data import folds
from scripts.data.folds import Split
from tests.data.test_loaders import write_files
from tests.helpers import make_config

NOW = dt.datetime(2026, 11, 13, tzinfo=dt.timezone.utc)


def facts(n=13, dbs_every=3, dysk_every=4):
    """{P01..: (DBS, dyskinetic)}: every 3rd patient without DBS, every 4th with dyskinetic hours."""
    return {f"P{i:02d}": (i % dbs_every != 0, i % dysk_every == 0) for i in range(1, n + 1)}


def test_folds_cover_everyone_once_spread_both_groups_and_are_reproducible():
    patients = facts()
    a = folds.make_folds(patients, 5, seed=42)
    assert sorted(p for f in a for p in f) == sorted(patients)
    assert max(map(len, a)) - min(map(len, a)) <= 1
    for column in (0, 1):                      # DBS, dyskinetic: each spread as evenly as possible
        for value in (True, False):
            per_fold = [sum(patients[p][column] == value for p in f) for f in a]
            assert max(per_fold) - min(per_fold) <= 1
    assert folds.make_folds(dict(reversed(list(patients.items()))), 5, seed=42) == a     # order doesn't matter
    assert folds.make_folds(patients, 5, seed=7) != a


def test_on_random_patient_mixes_every_spread_is_at_most_one():
    rng = np.random.default_rng(0)
    for trial in range(150):
        n = int(rng.integers(5, 40))
        k = int(rng.integers(2, min(8, n) + 1))
        p_dbs, p_dysk = rng.random(2)
        patients = {f"P{i}": (bool(rng.random() < p_dbs), bool(rng.random() < p_dysk)) for i in range(n)}
        f = folds.make_folds(patients, k, seed=trial)
        assert max(map(len, f)) - min(map(len, f)) <= 1
        for column in (0, 1):
            for value in (True, False):
                per_fold = [sum(patients[p][column] == value for p in fold) for fold in f]
                assert max(per_fold) - min(per_fold) <= 1


def test_too_few_patients_or_folds_are_refused():
    with pytest.raises(ConfigError, match="can't fill"):
        folds.make_folds(facts(3), 5, 0)
    with pytest.raises(ConfigError, match="2 or more"):
        folds.make_folds(facts(), 1, 0)


def test_create_sets_the_test_patients_aside_and_folds_the_rest():
    split = folds.create(facts(), ["P02", "P05"], 5, 42, "a written rule", now=NOW)
    assert split.test == ("P02", "P05") and len(split.development) == 11 and len(split.folds) == 5
    for train, val in split.pairs():
        assert not set(train) & set(val) and set(train) | set(val) == set(split.development)
        assert not (set(train) | set(val)) & set(split.test)
    with pytest.raises(ConfigError, match="not in the data"):
        folds.create(facts(), ["P99"], 5, 42, "rule")
    with pytest.raises(ConfigError, match="--rule"):
        folds.create(facts(), ["P02"], 5, 42, " ")


def test_adding_patients_never_moves_anyone():
    split = folds.create(facts(13), ["P02", "P05"], 5, 42, "rule", now=NOW)
    bigger = folds.add(split, facts(16), test=["P15"], now=NOW)
    assert bigger.test == ("P02", "P05", "P15")                        # the old test patients stay
    assert set(bigger.development) == set(split.development) | {"P14", "P16"}
    assert bigger.history[-1] == {"at": NOW.isoformat(timespec="seconds"), "test_added": 1, "development_added": 2,
                                  "folds_remade": True}
    assert bigger.hash != split.hash
    # Only a new test patient: the folds stay exactly as they were.
    only_test = folds.add(split, facts(14), test=["P14"], now=NOW)
    assert only_test.folds == split.folds and not only_test.history[-1]["folds_remade"]
    # Nothing new: the very same split.
    assert folds.add(split, facts(13)) is split


def test_a_development_patient_never_moves_into_the_test_set():
    split = folds.create(facts(), ["P02"], 5, 42, "rule")
    with pytest.raises(ConfigError, match="never moves into"):
        folds.add(split, facts(14), test=["P03"])
    with pytest.raises(ConfigError, match="already test"):
        folds.add(split, facts(14), test=["P02"])
    with pytest.raises(ConfigError, match="not in the data"):
        folds.add(split, facts(14), test=["P99"])


def test_check_needs_the_data_and_the_split_to_name_the_same_patients():
    split = folds.create(facts(), ["P02"], 5, 42, "rule")
    split.check(sorted(facts()))
    with pytest.raises(SchemaError, match="aren't in the split.*folds add"):
        split.check(sorted(facts(14)))
    with pytest.raises(SchemaError, match="aren't in the data"):
        split.check(sorted(facts(12)))


@pytest.mark.parametrize("test, fold_lists, match", [
    (["A"], [["A", "B"], ["C"]], "named twice"),
    ([], [["A", "B"], ["B"]], "named twice"),
    ([], [["A", "B"]], "at least 2 folds"),
    ([], [["A"], []], "none of them empty"),
])
def test_a_broken_split_is_refused(test, fold_lists, match):
    with pytest.raises(SchemaError, match=match):
        Split(test, fold_lists, 0)


def test_the_file_round_trips_and_the_hash_names_only_who_is_where(tmp_path):
    split = folds.create(facts(), ["P02"], 5, 42, "rule", now=NOW)
    path = folds.write_split(split, tmp_path / "split.json")
    back = Split.from_json(json.loads(path.read_text(encoding="utf-8")))
    assert back == split and back.hash == split.hash and back.history == split.history
    from dataclasses import replace
    assert replace(split, rule="other", history=()).hash == split.hash
    with pytest.raises(SchemaError, match="format"):
        Split.from_json({**split.to_json(), "format_version": 2})
    with pytest.raises(SchemaError, match="list of lists"):
        Split.from_json({"format_version": 1, "test": []})                   # no folds
    with pytest.raises(SchemaError, match="list of lists"):
        Split.from_json({**split.to_json(), "test": "AB"})                   # not read as patients A and B
    with pytest.raises(SchemaError, match="bad field"):
        Split.from_json({"format_version": 1, "test": [], "folds": [["A"], ["B"]]})   # no seed


def test_the_command_creates_once_then_only_adds(tmp_path, capsys):
    paths = write_files(tmp_path, people=[f"COPS-{i}" for i in range(1, 8)])
    patients = pd.read_csv(paths["patients_csv"]).assign(dbs=[True, False] * 3 + [True])
    patients.to_csv(paths["patients_csv"], index=False)
    paths["split_file"] = str(tmp_path / "split.json")
    cfg = make_config(tmp_path, features_csv=paths["features_csv"], paths=paths, split={"cv_folds": 3, "seed": 1})
    config_file = tmp_path / "config.yaml"
    import yaml
    config_file.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if k != "resolved_paths"}), encoding="utf-8")
    run = ["--config", str(config_file)]
    assert folds.main(run + ["create", "--test", "COPS-1", "--rule", "for the test"]) == 0
    split = folds.read_split(cfg)
    assert split.test == ("COPS-1",) and len(split.development) == 6 and len(split.folds) == 3
    with pytest.raises(ConfigError, match="made once"):
        folds.main(run + ["create", "--test", "COPS-2", "--rule", "again"])
    assert folds.main(run + ["add"]) == 0 and "unchanged" in capsys.readouterr().out
    assert folds.read_split(cfg) == split


def test_reading_a_missing_split_says_how_to_make_one(tmp_path):
    cfg = {"resolved_paths": {"split_file": tmp_path / "none.json"}}
    with pytest.raises(SchemaError, match="folds create"):
        folds.read_split(cfg)
    with pytest.raises(ConfigError, match="split_file"):
        folds.read_split({"resolved_paths": {}})


def test_patient_facts_reads_dbs_and_dyskinetic_hours():
    from scripts.core.data import Dataset, WindowSpec
    hours = pd.DataFrame({"participant_id": ["A", "A", "B"], "hour_index": [0, 1, 0]})
    labels = pd.DataFrame({"state_now": [3.0, 5.0, np.nan]})
    patients = pd.DataFrame({"participant_id": ["A", "B"], "more_affected_side": True, "dbs": [True, False]})
    data = Dataset(WindowSpec(1, 10, 30, 3), hours, labels, patients=patients)
    assert folds.patient_facts(data) == {"A": (True, True), "B": (False, False)}


def test_patient_facts_refuses_dbs_that_isnt_true_or_false():
    from scripts.core.data import Dataset, WindowSpec
    hours = pd.DataFrame({"participant_id": ["A", "B"], "hour_index": [0, 0]})
    labels = pd.DataFrame({"state_now": [3.0, 3.0]})
    for bad in ("no", np.nan, "False"):
        patients = pd.DataFrame({"participant_id": ["A", "B"], "more_affected_side": True, "dbs": [True, bad]})
        with pytest.raises(SchemaError, match="dbs"):
            folds.patient_facts(Dataset(WindowSpec(1, 10, 30, 3), hours, labels, patients=patients))


def test_too_few_patients_without_dbs_for_the_folds_is_warned(caplog):
    patients = {f"P{i}": (i > 1, False) for i in range(1, 9)}          # one patient without DBS, 4 folds
    with caplog.at_level("WARNING"):
        folds.make_folds(patients, 4, seed=0)
    assert "without DBS" in caplog.text
