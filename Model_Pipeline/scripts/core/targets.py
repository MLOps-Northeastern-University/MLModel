"""What the model predicts: the patient's state and their tremor, now and in the next hour.

Owner: P2.

Each target is defined once here, so a definition (such as what counts as OFF)
changes in one place. See MODEL.md, Decisions: Outputs, and How OFF is defined.

    state    the diary's kinesia score -3 to +3, or asleep: 8 classes
    tremor   the diary's tremor score 0 to 2: 3 classes

Both are classification over ordered scores, each predicted as one softmax.
The rest is worked out from the class probabilities:

    P(off)           sum of the STATE_GROUPS["off"] classes (scores -3 to -1, akinesia)
    expected score   sum of score x probability, using KINESIA_SCORES or TREMOR_SCORES
    P(any tremor)    P(1) + P(2)

A label is a target at a horizon: state_now is this hour's state, state_next_hour
the next hour's. Metric names use it: val/state_now/off_auc.

Labels are floats: the class index, or NaN where there's no answer ("No Data",
unanswered, tremor in a sleep hour, or no next hour in the table). NaN rows are
left out of training and scoring, never counted as 0.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import pandas as pd

from scripts.core.errors import ConfigError, SchemaError

# State classes: the 7 kinesia scores in order, then asleep. Index = score + 3.
STATE_CLASSES = (
    "severe_akinesia", "discomforting_akinesia", "slight_akinesia", "good_kinesia",
    "slight_dyskinesia", "discomforting_dyskinesia", "severe_dyskinesia", "asleep")
KINESIA_SCORES = (-3, -2, -1, 0, 1, 2, 3)
ASLEEP = 7
# The broad states, as the state classes whose probabilities add up to them.
STATE_GROUPS = {"off": (0, 1, 2), "on": (3,), "dyskinetic": (4, 5, 6), "asleep": (ASLEEP,)}

# Tremor classes. Index = score.
TREMOR_CLASSES = ("none", "weak_or_slight", "strong_or_severe")
TREMOR_SCORES = (0, 1, 2)

# How many hours ahead each horizon's label is.
HORIZONS = {"now": 0, "next_hour": 1}

# Columns of the data pipeline's hourly table.
KINESIA_TEXT = "kinesia_text"
KINESIA_SCORE = "kinesia_score"
TREMOR_SCORE = "tremor_score"
PARTICIPANT = "participant_id"
HOUR_INDEX = "hour_index"   # hours since the participant's first diary day began
SLEEP = "Sleep"             # the diary's kinesia answer for a sleep hour


@dataclass(frozen=True)
class Target:
    """One thing the model predicts.

    Attributes:
        name: Used in label names, configs and outputs.
        classes: The class names; a label is the index of one of them.
        scores: The diary score of each class, in class order; None for a class with
            no score (asleep). Used for the expected score.
        source_columns: The hourly table's columns the label is read from.
        derive: Turns the table into class indices (NaN for no label).
    """

    name: str
    classes: Tuple[str, ...]
    scores: Tuple[Optional[int], ...]
    source_columns: Tuple[str, ...]
    derive: Callable[[pd.DataFrame], pd.Series]

    def __post_init__(self):
        if len(self.scores) != len(self.classes):
            raise ConfigError(f"target {self.name!r}: one score per class needed")

    def labels(self, table):
        """This target's label for every row of the table, for the hour itself.

        Raises:
            SchemaError: A source column is missing, or a score isn't valid.
        """
        _require(table, self.source_columns, f"target {self.name!r}")
        return self.derive(table).astype(float).rename(self.name)


def _require(table, columns, needed_by):
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise SchemaError(f"{needed_by} needs column(s) {missing}")


def _read_scores(table, column, allowed):
    """The column as numbers, checking every score is one of allowed (NaN is fine)."""
    try:
        score = pd.to_numeric(table[column])
    except (ValueError, TypeError):
        raise SchemaError(f"{column} has a value that isn't a number") from None
    bad = score.notna() & ~score.isin(allowed)
    if bad.any():
        raise SchemaError(f"{column} must be a whole number from {allowed[0]} to {allowed[-1]}, "
                          f"not {score[bad].iloc[0]}")
    return score


def _state(table):
    score = _read_scores(table, KINESIA_SCORE, KINESIA_SCORES)
    state = score + 3   # score -3..+3 -> class 0..6
    # Sleep hours have no score; they're read from the diary's answer.
    return state.mask(table[KINESIA_TEXT] == SLEEP, ASLEEP)


def _tremor(table):
    score = _read_scores(table, TREMOR_SCORE, TREMOR_SCORES)
    # Not labelled in sleep hours: tremor while asleep is trivially none.
    return score.mask(table[KINESIA_TEXT] == SLEEP)


TARGETS = {
    "state": Target("state", STATE_CLASSES, KINESIA_SCORES + (None,), (KINESIA_TEXT, KINESIA_SCORE), _state),
    "tremor": Target("tremor", TREMOR_CLASSES, TREMOR_SCORES, (KINESIA_TEXT, TREMOR_SCORE), _tremor),
}


def label_name(target, horizon):
    """The name of a target at a horizon, such as state_now."""
    return f"{target}_{horizon}"


# Every label the pipeline can train, score and log metrics for.
LABEL_NAMES = frozenset(label_name(t, h) for t in TARGETS for h in HORIZONS)


def parse_label(label):
    """Split a label into its target and horizon: "state_next_hour" -> ("state", "next_hour").

    Raises:
        ConfigError: Not a known label.
    """
    for target in TARGETS:
        horizon = label[len(target) + 1:] if isinstance(label, str) and label.startswith(target + "_") else None
        if horizon in HORIZONS:
            return target, horizon
    raise ConfigError(f"unknown label {label!r}; known: {sorted(LABEL_NAMES)}")


def get_target(name):
    """Return the Target called name.

    Raises:
        ConfigError: No target has that name.
    """
    if name not in TARGETS:
        raise ConfigError(f"unknown target {name!r}; known: {sorted(TARGETS)}")
    return TARGETS[name]


def at_horizon(table, y, horizon):
    """Give each row the label from `horizon` hours later, for the same participant.

    Args:
        table: The table y was derived from, with participant_id and hour_index.
        y: Labels for the hour itself, on the table's index.
        horizon: A key of HORIZONS.

    Returns:
        pd.Series on the table's index. NaN where that later hour isn't in the
        table, so a gap or the end of a recording never borrows another hour's label.

    Raises:
        ConfigError: Unknown horizon.
        SchemaError: The columns are missing, or a participant has an hour twice.
    """
    if horizon not in HORIZONS:
        raise ConfigError(f"unknown horizon {horizon!r}; known: {sorted(HORIZONS)}")
    hours = HORIZONS[horizon]
    if hours == 0:
        return y
    _require(table, (PARTICIPANT, HOUR_INDEX), f"horizon {horizon!r}")

    # Every label, looked up by (participant, hour).
    by_hour = pd.Series(y.to_numpy(), index=pd.MultiIndex.from_arrays([table[PARTICIPANT], table[HOUR_INDEX]]))
    if by_hour.index.duplicated().any():
        raise SchemaError(f"a participant has the same {HOUR_INDEX} twice")
    # For each row, fetch (same participant, hour + hours); missing ones become NaN.
    later = by_hour.reindex(pd.MultiIndex.from_arrays([table[PARTICIPANT], table[HOUR_INDEX] + hours]))
    return pd.Series(later.to_numpy(), index=table.index, dtype=float)


def build_labels(table, targets, horizons=tuple(HORIZONS)):
    """One label column per target and horizon, such as state_now and state_next_hour.

    Args:
        table: The data pipeline's hourly table.
        targets: Target names, such as ["state", "tremor"].
        horizons: Keys of HORIZONS. Defaults to all of them.

    Returns:
        pd.DataFrame on the table's index, columns in the order target, then horizon.

    Raises:
        ConfigError: No targets or horizons, or an unknown one.
    """
    if not targets or not horizons:
        raise ConfigError("name at least one target and one horizon")
    columns = {}
    for name in targets:
        y = get_target(name).labels(table)
        for horizon in horizons:
            columns[label_name(name, horizon)] = at_horizon(table, y, horizon)
    return pd.DataFrame(columns, index=table.index)
