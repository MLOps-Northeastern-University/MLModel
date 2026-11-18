"""The patient split: the fixed test patients, and folds by patient over the rest.

Owner: P1 (component B). Written by P2 as the stand-in train.py needs; P1 takes it over.

The test patients are set aside once, before any model work, and never used for training, tuning,
choosing epochs or any statistic. So the split is a file, made once and only ever added to, never
recomputed. If it were computed from the patient list (a seeded shuffle, a fraction as test), adding
patients would reshuffle it and move patients the model was tuned on into the test set.

    the split file (config paths.split_file), kept with the data and versioned by DVC, never in Git,
    because it names patients:

        {"format_version": 1, "seed": 42, "test": [...], "folds": [[...], [...], ...],
         "rule": "how the test patients were chosen", "history": [what each change did, and when]}

A pipeline run only reads it:

    split = read_split(cfg)
    split.check(data.participants)        # the data and the split name exactly the same patients
    for train_ids, val_ids in split.pairs(): ...
    split.hash                            # logged with every run: CV scores compare only within one split

Changing it is a command, run on purpose and never by the pipeline:

    python -m model_pipeline.data.folds create --test COPS-38 COPS-45 --rule "why these"
    python -m model_pipeline.data.folds add                       # new patients become development patients
    python -m model_pipeline.data.folds add --test COPS-70        # a new patient joins the test set

The rules it enforces: no one ever leaves the test set, and no development patient ever moves into
it, because they have already shaped choices. Which new patients join the test set is a person's
decision, named on the command line, never a random draw. With this few patients a random test
patient can be useless (no OFF hours, say), so the choice follows a written rule (the file's "rule").

The folds are remade whenever the development patients change, by patient and with the file's seed.
Patients with and without DBS, and with and without dyskinetic hours, are each spread over the folds
as evenly as possible (make_folds), so every fold can score OFF in both DBS groups, and dyskinesia.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from model_pipeline.core.errors import ConfigError, SchemaError
from model_pipeline.core.targets import PARTICIPANT, STATE_GROUPS
from model_pipeline.utils.logging_setup import get_logger

logger = get_logger(__name__)

FORMAT_VERSION = 1
FOLD_TRIES = 200                   # seeded orders make_folds tries (about 16 ms for 60 patients)


@dataclass(frozen=True)
class Split:
    """The test patients and the folds over the development patients.

    Raises:
        SchemaError: A patient named twice, in both test and a fold, an empty fold, or fewer than 2 folds.
    """

    test: Tuple[str, ...]
    folds: Tuple[Tuple[str, ...], ...]
    seed: int
    rule: str = ""
    history: Tuple[Mapping, ...] = field(default=(), compare=False)

    def __post_init__(self):
        object.__setattr__(self, "test", tuple(sorted(self.test)))
        object.__setattr__(self, "folds", tuple(tuple(sorted(f)) for f in self.folds))
        object.__setattr__(self, "history", tuple(self.history))
        named = list(self.test) + [p for f in self.folds for p in f]
        if not all(isinstance(p, str) and p for p in named):
            raise SchemaError("the split's patients must be non-empty names")
        if len(named) != len(set(named)):
            raise SchemaError("a patient is named twice in the split (in test and a fold, or in two folds)")
        if len(self.folds) < 2 or not all(self.folds):
            raise SchemaError("the split needs at least 2 folds, none of them empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise SchemaError(f"the split's seed must be a non-negative whole number, not {self.seed!r}")

    @property
    def development(self) -> Tuple[str, ...]:
        """Every patient in a fold, sorted."""
        return tuple(sorted(p for f in self.folds for p in f))

    @property
    def patients(self) -> Tuple[str, ...]:
        return tuple(sorted(self.test + self.development))

    @property
    def hash(self) -> str:
        """Names this exact split (who is in test and in each fold); the seed, rule and history don't count."""
        text = json.dumps({"test": list(self.test), "folds": [list(f) for f in self.folds]}, separators=(",", ":"))
        return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()

    def pairs(self):
        """(training patients, validation patients) for each fold, in fold order. Test is in neither."""
        return [(tuple(p for p in self.development if p not in fold), fold) for fold in self.folds]

    def check(self, participants: Sequence[str]) -> None:
        """Raise SchemaError unless participants are exactly the split's patients."""
        given, known = set(participants), set(self.patients)
        unknown, missing = sorted(given - known), sorted(known - given)
        if unknown:
            raise SchemaError(f"{len(unknown)} patient(s) in the data aren't in the split, e.g. {unknown[0]}: "
                              "add them with python -m model_pipeline.data.folds add")
        if missing:
            raise SchemaError(f"{len(missing)} patient(s) in the split aren't in the data, e.g. {missing[0]}")

    def to_json(self) -> Dict:
        return {"format_version": FORMAT_VERSION, "seed": self.seed, "test": list(self.test),
                "folds": [list(f) for f in self.folds], "rule": self.rule, "history": list(self.history)}

    @classmethod
    def from_json(cls, obj: Mapping) -> "Split":
        if obj.get("format_version") != FORMAT_VERSION:
            raise SchemaError(f"split file format {obj.get('format_version')!r}, this code reads {FORMAT_VERSION}")
        test, folds = obj.get("test"), obj.get("folds")
        if not (isinstance(test, list) and isinstance(folds, list) and all(isinstance(f, list) for f in folds)):
            raise SchemaError("the split file's test must be a list of patients and folds a list of lists")
        try:
            return cls(obj["test"], obj["folds"], obj["seed"], obj.get("rule", ""), obj.get("history", ()))
        except (KeyError, TypeError) as e:
            raise SchemaError(f"the split file is missing or has a bad field: {e}") from None


# ---- making and changing a split ---------------------------------------------------------------

def make_folds(patients: Mapping[str, Tuple[bool, bool]], n_folds: int, seed: int) -> Tuple[Tuple[str, ...], ...]:
    """Folds by patient, with DBS patients and dyskinetic patients each spread over them.

    Args:
        patients: {patient: (has DBS, has any dyskinetic hour)} for the development patients.
        n_folds: How many folds; at most one per patient.
        seed: Makes the folds reproducible: the same patients and seed always give the same folds.

    One pass deals the patients out in a random order, each to the fold with the fewest patients of
    its DBS status, then the fewest of its dyskinetic status, then the fewest patients. A single pass
    can still bunch one group, so FOLD_TRIES seeded orders are tried and the most even result is kept:
    the smallest spread (largest fold minus smallest) of fold sizes, then of DBS, then of dyskinetic.
    On 1,500 random patient mixes (5 to 60 patients, 2 to 10 folds) every spread was at most one.
    """
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2:
        raise ConfigError(f"the number of folds must be a whole number of 2 or more, not {n_folds!r}")
    if len(patients) < n_folds:
        raise ConfigError(f"{len(patients)} development patients can't fill {n_folds} folds")
    no_dbs = sum(not bool(v[0]) for v in patients.values())
    if 0 < no_dbs < n_folds:
        logger.warning("only %d development patients without DBS for %d folds: %d fold(s) can't score OFF for "
                       "patients without DBS", no_dbs, n_folds, n_folds - no_dbs,
                       extra={"event": "folds_short_of_no_dbs", "no_dbs": no_dbs, "folds": n_folds})
    rng = np.random.default_rng(seed)
    ids = sorted(patients)                   # so the input's order never changes the folds
    facts = {p: (int(bool(patients[p][0])), int(bool(patients[p][1]))) for p in ids}
    best = None
    for _ in range(FOLD_TRIES):
        folds, dbs, dyskinetic = _deal(rng.permutation(ids), facts, n_folds)
        sizes = [len(f) for f in folds]
        spread = (max(sizes) - min(sizes), int((dbs.max(0) - dbs.min(0)).max()),
                  int((dyskinetic.max(0) - dyskinetic.min(0)).max()))
        if best is None or spread < best[0]:
            best = (spread, folds)
    return tuple(tuple(str(p) for p in f) for f in best[1])


def _deal(order, facts, n_folds):
    """One pass of make_folds: the folds, and per fold the count of each DBS and dyskinetic status."""
    folds = [[] for _ in range(n_folds)]
    dbs, dyskinetic = np.zeros((n_folds, 2), int), np.zeros((n_folds, 2), int)
    for pid in order:
        d, y = facts[pid]
        f = min(range(n_folds), key=lambda i: (dbs[i, d], dyskinetic[i, y], len(folds[i]), i))
        folds[f].append(pid)
        dbs[f, d] += 1
        dyskinetic[f, y] += 1
    return folds, dbs, dyskinetic


def create(patients: Mapping[str, Tuple[bool, bool]], test: Sequence[str], n_folds: int, seed: int,
           rule: str, now=None) -> Split:
    """A new split: these test patients, and folds over everyone else."""
    test = set(test)
    unknown = sorted(test - set(patients))
    if unknown:
        raise ConfigError(f"test patient(s) not in the data: {unknown}")
    if not rule.strip():
        raise ConfigError("say how the test patients were chosen (--rule): the split records it")
    development = {p: v for p, v in patients.items() if p not in test}
    folds = make_folds(development, n_folds, seed)
    return Split(tuple(test), folds, seed, rule, ({"at": _now(now), "created": True, "test": len(test),
                                                   "development": len(development), "folds": n_folds},))


def add(split: Split, patients: Mapping[str, Tuple[bool, bool]], test: Sequence[str] = (), now=None) -> Split:
    """The split with the data's new patients added: those named in test to the test set, the rest to
    development (which remakes the folds). Existing patients never move.

    Args:
        patients: {patient: (has DBS, has any dyskinetic hour)} for every patient in the data.
        test: New patients that join the test set.

    Raises:
        ConfigError: A test patient who is already in the split, or isn't in the data.
    """
    test = set(test)
    already = sorted(test & set(split.development))
    if already:
        raise ConfigError(f"{already} already train or validate models; a development patient never moves into "
                          "the test set")
    if test & set(split.test):
        raise ConfigError(f"{sorted(test & set(split.test))} are already test patients")
    unknown = sorted(test - set(patients))
    if unknown:
        raise ConfigError(f"test patient(s) not in the data: {unknown}")
    new = set(patients) - set(split.patients)
    if not new:
        return split
    new_development = sorted(new - test)
    folds = split.folds
    if new_development:
        everyone = set(split.development) | set(new_development)
        missing = sorted(everyone - set(patients))
        if missing:
            raise ConfigError(f"development patient(s) not in the data, so the folds can't be remade: {missing[:3]}")
        folds = make_folds({p: patients[p] for p in everyone}, len(split.folds), split.seed)
    change = {"at": _now(now), "test_added": len(test), "development_added": len(new_development),
              "folds_remade": bool(new_development)}
    return replace(split, test=split.test + tuple(sorted(test)), folds=folds, history=split.history + (change,))


def _now(now):
    return (now or dt.datetime.now(dt.timezone.utc)).isoformat(timespec="seconds")


# ---- reading and writing -------------------------------------------------------------------------

def split_path(cfg) -> Path:
    try:
        return Path(cfg["resolved_paths"]["split_file"])
    except KeyError:
        raise ConfigError("config paths needs split_file: where the patient split is kept") from None


def read_split(cfg) -> Split:
    """The split from paths.split_file.

    Raises:
        SchemaError: No split file yet, or one this code can't read.
    """
    path = split_path(cfg)
    if not path.exists():
        raise SchemaError(f"no patient split at {path}: make it once with python -m model_pipeline.data.folds create")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise SchemaError(f"{path} isn't valid JSON: {e}") from None
    split = Split.from_json(obj)
    logger.info("patient split %s: %d test, %d development in %d folds", split.hash, len(split.test),
                len(split.development), len(split.folds), extra={
                    "event": "split_read", "split_hash": split.hash, "test": len(split.test),
                    "development": len(split.development), "folds": len(split.folds)})
    return split


def write_split(split: Split, path) -> Path:
    """Write the split, through a temporary file, so a crash never leaves half a split."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(split.to_json(), indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def patient_facts(data) -> Dict[str, Tuple[bool, bool]]:
    """{patient: (has DBS, has any dyskinetic hour)}, from a Dataset with the patients group."""
    if data.patients is None:
        raise SchemaError("the split needs the patients table (DBS)")
    data.check(("patients",), data.spec)      # dbs must be True or False: "no" or NaN would read as True
    dbs = dict(zip(data.patients[PARTICIPANT], data.patients["dbs"].astype(bool)))
    state = data.labels["state_now"].to_numpy()
    dyskinetic = np.isin(state, list(STATE_GROUPS["dyskinetic"]))
    has = dict(zip(*np.unique(data.hours[PARTICIPANT].to_numpy()[dyskinetic], return_counts=True)))
    missing = sorted(set(data.participants) - set(dbs))
    if missing:
        raise SchemaError(f"{len(missing)} patient(s) have no row in the patients table, e.g. {missing[0]}")
    return {p: (bool(dbs[p]), bool(has.get(p, 0))) for p in data.participants}


# ---- the command -----------------------------------------------------------------------------------

def main(argv=None):
    from model_pipeline.data.loaders import load_dataset
    from model_pipeline.utils.config import load_config
    from model_pipeline.utils.logging_setup import configure_logging

    parser = argparse.ArgumentParser(description="Make or add to the patient split (see model_pipeline/data/folds.py)")
    parser.add_argument("--config", default=None, help="Path to a config YAML")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("create", help="make the split once")
    make.add_argument("--test", nargs="+", required=True, help="the test patients")
    make.add_argument("--rule", required=True, help="how the test patients were chosen")
    more = commands.add_parser("add", help="add the data's new patients")
    more.add_argument("--test", nargs="*", default=[], help="new patients who join the test set")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    configure_logging(cfg)
    path = split_path(cfg)
    patients = patient_facts(load_dataset(cfg, groups=("patients",)))
    if args.command == "create":
        if path.exists():
            raise ConfigError(f"{path} already exists; a split is made once. Use add for new patients.")
        split = create(patients, args.test, cfg["split"]["cv_folds"], cfg["split"]["seed"], args.rule)
    else:
        old = read_split(cfg)
        split = add(old, patients, args.test)
        if split is old:
            print(f"no new patients; the split is unchanged ({old.hash})")
            return 0
    write_split(split, path)
    print(f"split {split.hash}: {len(split.test)} test, {len(split.development)} development in "
          f"{len(split.folds)} folds -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
