import pytest

from scripts.core.data import WindowSpec
from scripts.core.errors import ConfigError
from scripts.data.folds import make_folds, split
from scripts.utils.config import load_config
from tests.helpers import make_dataset

TEN = [f"COPS-{i}" for i in (3, 4, 11, 18, 24, 25, 42, 44, 46, 60)]


@pytest.fixture
def cfg():
    return load_config()


def test_the_configured_folds_test_everyone_once(cfg):
    folds = make_folds(TEN, cfg)
    assert len(folds) == 5
    assert sorted(p for f in folds for p in f.test) == sorted(TEN)
    for f in folds:
        parts = [set(f.train), set(f.validation), set(f.test)]
        assert set.union(*parts) == set(TEN) and sum(map(len, parts)) == 10  # no one in two parts
        assert (len(f.train), len(f.validation), len(f.test)) == (7, 1, 2)
    assert len({f.validation for f in folds}) == 5  # a different validation participant each time


def test_the_same_config_gives_the_same_folds(cfg):
    assert make_folds(TEN, cfg) == make_folds(list(reversed(TEN)), cfg)


def fold(test, validation):
    return {"test": test, "validation": validation}


@pytest.mark.parametrize("folds, match", [
    ([fold(["A", "B"], "C"), fold(["B", "C"], "A")], "more than one fold"),
    ([fold(["A", "B"], "C")], "not tested \\['C'\\]"),
    ([fold(["A", "B"], "C"), fold(["C", "D"], "A")], "not in the data \\['D'\\]"),
    ([fold(["A"], "A"), fold(["B", "C"], "A")], "validation participant A"),
    ([fold(["A", "B"], "C"), fold(["C"], "A")], "fold 1 leaves no one"),
])
def test_bad_folds_are_refused(cfg, folds, match):
    cfg["split"]["folds"] = folds
    with pytest.raises(ConfigError, match=match):
        make_folds(["A", "B", "C"], cfg)


def test_split_gives_each_part_its_own_participants(cfg):
    data = make_dataset(WindowSpec(1, 10, 30, 3), n_participants=4)
    cfg["split"]["folds"] = [fold([f"COPS-{i}"], f"COPS-{i % 4 + 1}") for i in range(1, 5)]
    first = make_folds(data.participants, cfg)[0]
    train, validation, test = split(data, first)
    assert test.participants == ("COPS-1",)
    assert validation.participants == ("COPS-2",) and train.participants == ("COPS-3", "COPS-4")
