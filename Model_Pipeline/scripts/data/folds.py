"""Splits a Dataset by participant into the folds of the evaluation protocol.

Owner: P1 (component B).

The folds are fixed by hand in config split.folds: each names its test
participants and its validation participant (early stopping); everyone else in
the data trains. Every participant is tested exactly once, and no one is in two
parts of a fold, so the same config gives the same folds on every machine.
"""

from dataclasses import dataclass
from typing import Tuple

from scripts.core.errors import ConfigError


@dataclass(frozen=True)
class Fold:
    number: int
    train: Tuple[str, ...]
    validation: Tuple[str, ...]
    test: Tuple[str, ...]


def make_folds(participants, cfg):
    """The folds for these participants, from config split.folds.

    Raises:
        ConfigError: A participant is tested twice or never, a fold names someone
            not in the data, or a fold's validation participant is also tested or
            leaves no one to train on.
    """
    everyone = set(participants)
    tested = [p for f in cfg["split"]["folds"] for p in f["test"]]
    if len(tested) != len(set(tested)):
        raise ConfigError("split.folds: a participant is tested in more than one fold")
    if set(tested) != everyone:
        raise ConfigError(f"split.folds must test every participant once: not tested {sorted(everyone - set(tested))}, "
                          f"not in the data {sorted(set(tested) - everyone)}")
    out = []
    for i, f in enumerate(cfg["split"]["folds"], start=1):
        test, validation = set(f["test"]), f["validation"]
        if validation not in everyone - test:
            raise ConfigError(f"split.folds: fold {i}'s validation participant {validation} must be in the data "
                              "and not tested in that fold")
        train = tuple(sorted(everyone - test - {validation}))
        if not train:
            raise ConfigError(f"split.folds: fold {i} leaves no one to train on")
        out.append(Fold(i, train, (validation,), tuple(sorted(test))))
    return out


def split(dataset, fold):
    """The fold's (train, validation, test) Datasets."""
    return tuple(dataset.subset(part) for part in (fold.train, fold.validation, fold.test))
