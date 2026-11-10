"""What the model can predict: OFF, tremor and dyskinesia, each defined once.

Owner: P2.

Each target records its kind (binary probability or an ordinal score), its range,
whether it carries an uncertainty interval, and how its hourly label is derived
from the data pipeline's table. Metrics, plots, bias checks and the export read
targets from here, so a definition (such as what counts as OFF) changes in one
place. See MODEL.md, section 2 and Decisions.

OFF follows clinical practice: the patient's diary says akinesia (the COPS
kinesia score -3 to -1, which the data pipeline maps to kinesia_class "off").
Tremor is its own target, not the definition of OFF.

Labels are floats: 1/0 for binary targets, the score for ordinal ones, and NaN
where the diary has no usable answer for that hour (asleep, no data, a value
the data pipeline couldn't reconcile). Rows with a NaN label are left out of
that target's training and scoring, never counted as 0.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

from scripts.core.errors import ConfigError, SchemaError

BINARY = "binary"
ORDINAL = "ordinal"

KINESIA_CLASS = "kinesia_class"
TREMOR_SCORE = "tremor_score"


@dataclass(frozen=True)
class Target:
    """One thing the model predicts.

    Attributes:
        name: Used in configs, metric names ({split}/{target}/{metric}) and outputs.
        kind: BINARY (a probability from 0 to 1) or ORDINAL (a score in value_range).
        value_range: The lowest and highest label value.
        has_interval: Whether predictions carry a low and high bound.
        source_columns: Columns of the data pipeline's table the label is read from.
        description: What a positive or high value means, in plain words.
        derive: Turns the table's source columns into the label Series.
    """

    name: str
    kind: str
    value_range: Tuple[float, float]
    has_interval: bool
    source_columns: Tuple[str, ...]
    description: str
    derive: Callable[[pd.DataFrame], pd.Series]

    def labels(self, table):
        """Return this target's hourly label for every row of the table.

        Args:
            table: The data pipeline's hourly table, or any DataFrame with
                this target's source columns.

        Returns:
            pd.Series of floats on the table's index; NaN where there's no label.

        Raises:
            SchemaError: A source column is missing, or a label is out of range.
        """
        missing = [c for c in self.source_columns if c not in table.columns]
        if missing:
            raise SchemaError(f"target {self.name!r} needs column(s) {missing}")
        y = self.derive(table).astype(float).rename(self.name)
        low, high = self.value_range
        bad = y.notna() & ~y.between(low, high)
        if bad.any():
            raise SchemaError(
                f"target {self.name!r}: {int(bad.sum())} label(s) outside "
                f"{low}..{high}, e.g. {y[bad].iloc[0]}")
        return y


def _kinesia_is(label):
    """Label 1 where kinesia_class equals label, 0 for another class, NaN if none."""
    def derive(table):
        cls = table[KINESIA_CLASS]
        return pd.Series(np.where(cls.isna(), np.nan, (cls == label).astype(float)),
                         index=table.index)
    return derive


def _tremor_score(table):
    return pd.to_numeric(table[TREMOR_SCORE], errors="coerce")


TARGETS: Dict[str, Target] = {
    t.name: t for t in (
        Target(
            name="off",
            kind=BINARY,
            value_range=(0.0, 1.0),
            has_interval=False,
            source_columns=(KINESIA_CLASS,),
            description="The diary says akinesia (slight, discomforting or severe): "
                        "the medicine isn't working.",
            derive=_kinesia_is("off"),
        ),
        Target(
            name="dyskinesia",
            kind=BINARY,
            value_range=(0.0, 1.0),
            has_interval=False,
            source_columns=(KINESIA_CLASS,),
            description="The diary says dyskinesia (slight, discomforting or severe): "
                        "extra, involuntary movement.",
            derive=_kinesia_is("dyskinetic"),
        ),
        Target(
            name="tremor",
            kind=ORDINAL,
            value_range=(0.0, 2.0),
            has_interval=True,
            source_columns=(TREMOR_SCORE,),
            description="The diary's tremor score: 0 none, 1 weak or slight, "
                        "2 strong or severe.",
            derive=_tremor_score,
        ),
    )
}


def get_target(name):
    """Return the Target called name.

    Raises:
        ConfigError: No target has that name.
    """
    try:
        return TARGETS[name]
    except KeyError:
        raise ConfigError(f"unknown target {name!r}; known: {sorted(TARGETS)}") from None


def build_labels(table, names):
    """Return a DataFrame with one label column per named target.

    Args:
        table: The data pipeline's hourly table.
        names: Target names, such as ["off", "tremor"].

    Returns:
        pd.DataFrame on the table's index, one float column per target.
    """
    if not names:
        raise ConfigError("no targets named")
    return pd.concat([get_target(n).labels(table) for n in names], axis=1)
