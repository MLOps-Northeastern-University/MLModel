"""What the model predicts: the patient's state and their tremor, now and in the next hour.

Owner: P2.

Each target is defined once here: its classes, the diary columns it's read from,
and how its hourly label is derived. Metrics, plots, bias checks and the export
read targets from here, so a definition (such as what counts as OFF) changes in
one place. See MODEL.md, Decisions: Outputs, and How OFF is defined.

    state        one of asleep, off, on, dyskinetic, from the diary's kinesia answer
    tremor_ge1   tremor 1 or 2 (weak to severe), from the diary's tremor answer
    tremor_ge2   tremor 2 (strong or severe)

OFF follows clinical practice: the diary says akinesia (kinesia score -3 to -1,
which the data pipeline maps to kinesia_class "off"). Tremor is its own target,
not the definition of OFF.

Every target is predicted at each horizon, and a target at a horizon is a label:
state_now is this hour's state, state_next_hour the next hour's. Metric names use
the label: val/state_now/off_auc.

Labels are floats: the index of the class (0, 1, ...), or NaN where there's no
usable answer. NaN rows are left out of that label's training and scoring, never
counted as 0. There's no answer when the diary says "No Data" or the question
wasn't answered; for tremor, also in sleep hours (tremor while asleep is
trivially none, and the asleep state covers it); for next_hour, also when the
participant's next hour isn't in the table.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import pandas as pd

from scripts.core.errors import ConfigError, SchemaError

STATES = ("asleep", "off", "on", "dyskinetic")
NO_YES = ("no", "yes")
# How many hours ahead each horizon's label is.
HORIZONS = {"now": 0, "next_hour": 1}

KINESIA_TEXT = "kinesia_text"
KINESIA_CLASS = "kinesia_class"
TREMOR_SCORE = "tremor_score"
PARTICIPANT = "participant_id"
HOUR_INDEX = "hour_index"   # hours since the participant's first diary day began
SLEEP = "Sleep"             # the diary's kinesia answer for a sleep hour
TREMOR_MAX = 2


@dataclass(frozen=True)
class Target:
    """One thing the model predicts, as a choice between classes.

    Attributes:
        name: Used in label names, configs and outputs.
        classes: The class names; a label is the index of one of them.
        source_columns: Columns of the data pipeline's table the label is read from.
        description: What the classes mean, in plain words.
        derive: Turns the table's source columns into class indices (NaN for no label).
    """

    name: str
    classes: Tuple[str, ...]
    source_columns: Tuple[str, ...]
    description: str
    derive: Callable[[pd.DataFrame], pd.Series]

    @property
    def is_binary(self):
        return len(self.classes) == 2

    def labels(self, table):
        """Return this target's label for every row of the table, for the hour itself.

        Args:
            table: The data pipeline's hourly table, or any DataFrame with
                this target's source columns.

        Returns:
            pd.Series of floats on the table's index: class indices, NaN for no label.

        Raises:
            SchemaError: A source column is missing, or a value can't be a label.
        """
        _require(table, self.source_columns, f"target {self.name!r}")
        y = self.derive(table).astype(float).rename(self.name)
        bad = y.notna() & ~y.isin(range(len(self.classes)))
        if bad.any():   # a derive bug, not bad data: derive raises on bad data itself
            raise SchemaError(f"target {self.name!r}: {y[bad].iloc[0]} isn't a class index")
        return y


def _require(table, columns, needed_by):
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise SchemaError(f"{needed_by} needs column(s) {missing}")


def _state(table):
    # Sleep hours have no kinesia_class (the data pipeline maps only the 7 kinesia
    # scores), so they're read from the diary's answer.
    state = table[KINESIA_CLASS].where(table[KINESIA_TEXT] != SLEEP, "asleep")
    unknown = state.notna() & ~state.isin(STATES)
    if unknown.any():
        raise SchemaError(f"unknown {KINESIA_CLASS} {state[unknown].iloc[0]!r}; known: {STATES[1:]}")
    return state.map({name: i for i, name in enumerate(STATES)})


def _tremor_at_least(level):
    """1 where the tremor score is level or more, 0 below it; NaN in sleep hours or with no score."""
    def derive(table):
        score = pd.to_numeric(table[TREMOR_SCORE], errors="coerce")
        bad = score.notna() & ~score.between(0, TREMOR_MAX)
        if bad.any():
            raise SchemaError(f"{TREMOR_SCORE} must be 0 to {TREMOR_MAX}, not {score[bad].iloc[0]}")
        score = score.mask(table[KINESIA_TEXT] == SLEEP)
        return (score >= level).astype(float).where(score.notna())
    return derive


TARGETS: Dict[str, Target] = {
    t.name: t for t in (
        Target(
            name="state",
            classes=STATES,
            source_columns=(KINESIA_TEXT, KINESIA_CLASS),
            description="asleep: the diary says Sleep. off: akinesia (slight, discomforting "
                        "or severe), the medicine isn't working. on: good kinesia. dyskinetic: "
                        "dyskinesia (slight, discomforting or severe), extra involuntary movement.",
            derive=_state,
        ),
        Target(
            name="tremor_ge1",
            classes=NO_YES,
            source_columns=(KINESIA_TEXT, TREMOR_SCORE),
            description="yes: the diary's tremor score is 1 or 2 (weak, slight, strong or severe).",
            derive=_tremor_at_least(1),
        ),
        Target(
            name="tremor_ge2",
            classes=NO_YES,
            source_columns=(KINESIA_TEXT, TREMOR_SCORE),
            description="yes: the diary's tremor score is 2 (strong or severe).",
            derive=_tremor_at_least(2),
        ),
    )
}


def label_name(target, horizon):
    """The name of a target at a horizon, such as state_now."""
    return f"{target}_{horizon}"


# Every label the pipeline can train, score and log metrics for.
LABEL_NAMES = frozenset(label_name(t, h) for t in TARGETS for h in HORIZONS)


def get_target(name):
    """Return the Target called name.

    Raises:
        ConfigError: No target has that name.
    """
    try:
        return TARGETS[name]
    except KeyError:
        raise ConfigError(f"unknown target {name!r}; known: {sorted(TARGETS)}") from None


def at_horizon(table, y, horizon):
    """Line each row up with the label `horizon` ahead, for the same participant.

    Args:
        table: The table y was derived from, with participant_id and hour_index.
        y: Labels for the hour itself, on the table's index.
        horizon: A key of HORIZONS.

    Returns:
        pd.Series on the table's index. NaN where the participant's hour ahead
        isn't in the table, so a gap or the end of a recording never borrows
        another hour's label.

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
    here = pd.MultiIndex.from_arrays([table[PARTICIPANT], table[HOUR_INDEX]])
    if here.duplicated().any():
        raise SchemaError(f"a participant has the same {HOUR_INDEX} twice, e.g. {here[here.duplicated()][0]}")
    ahead = pd.MultiIndex.from_arrays([table[PARTICIPANT], table[HOUR_INDEX] + hours])
    by_hour = pd.Series(y.to_numpy(), index=here)
    return pd.Series(by_hour.reindex(ahead).to_numpy(), index=table.index, dtype=float)


def build_labels(table, targets, horizons=tuple(HORIZONS)):
    """Return one label column per target and horizon, named as label_name gives.

    Args:
        table: The data pipeline's hourly table.
        targets: Target names, such as ["state", "tremor_ge1"].
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
