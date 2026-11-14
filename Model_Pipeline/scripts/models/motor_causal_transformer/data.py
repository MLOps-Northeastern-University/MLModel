"""From a Dataset to the network's batches: a recording per participant, chunks of it, batches.

    recordings = prepare(dataset, cache, labels, window_s=10, dose_span_s=1800)    # once per fit or predict
    for group in batches(tile_chunks(recordings, 8, rng), 4, rng):                # one epoch
        examples = [make_example(recordings[p], start, hours=8, past_h=1, ...) for p, start in group]
        collated = collate(examples, labels)                                       # Batch, graded, future

A recording is one participant's worn windows, in order: their times, harnet10 features (harnet.py),
event fields and patient fields, and the diary's answers per hour. The event fields are worked out
once for the whole recording and sliced (tokens.py says why). Only worn windows are tokens.

An example is a chunk: `hours` consecutive diary hours, plus `past_h` hours of windows in front of
them that the model reads but isn't graded on. The final recipe uses 8 h chunks with 1 h of past,
tiled over each recording from a random offset every epoch, so the chunk boundaries move.

Check-ins are an ordinary input, a few true answers scattered through the past. COPS has none, so
training simulates them. Each answered waking hour of the chunk, past included, gets one with
probability check_ins_per_day / 16 (about 16 waking hours a day). It goes on a random worn window
of the hour and carries that hour's diary answer: kinesia = state - 3, tremor as is. Real check-ins
in the data are already in the recording's event fields. The answers only ever come from the
recordings given, which in a fit are the training participants'.

Pretraining also needs `future`: the same windows run on future_h hours past the chunk, read only
to build its targets (pretraining.py). It holds raw features; the training loop standardises them.
"""

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch

from scripts.core.data import SECONDS_PER_HOUR, Dataset
from scripts.core.errors import ConfigError, ModelError
from scripts.core.targets import HOUR_INDEX, PARTICIPANT
from scripts.models.motor_causal_transformer.network import Batch
from scripts.models.motor_causal_transformer.tokens import event_fields, mark_check_ins, patient_fields
from scripts.utils.logging_setup import get_logger

logger = get_logger(__name__)

HOUR = SECONDS_PER_HOUR
WAKING_HOURS_PER_DAY = 16


@dataclass(frozen=True, eq=False)
class Recording:
    """One participant's worn windows, in order, and their diary answers.

    Attributes:
        participant: Its id.
        time_s: (n,) float64 each worn window's start on the diary clock, increasing.
        features: (n, 1024) float32 harnet10 features (memory-mapped from the cache).
        events: (n, 8) float32 the whole recording's event fields (tokens.EVENT_FIELDS).
        patient: (2,) float32 the patient fields (tokens.PATIENT_FIELDS).
        answers: {hour_index: (n_labels,) float32} class indices, NaN for no answer; only hours
            with at least one answer.
        labels: The label names answers' columns are, in order.
        weights: {hour_index: weight} for the diary step's loss and class balance (bias mitigation
            by reweighting), or None for 1 everywhere. Pretraining and validation don't use them.
    """

    participant: str
    time_s: np.ndarray
    features: np.ndarray
    events: np.ndarray
    patient: np.ndarray
    answers: Dict[int, np.ndarray]
    labels: Tuple[str, ...]
    weights: Optional[Dict[int, float]] = None

    def hours(self) -> Tuple[int, int]:
        """The first and last diary hour with a worn window."""
        if not len(self.time_s):
            raise ModelError(f"{self.participant} has no worn windows")
        return int(self.time_s[0] // HOUR), int(self.time_s[-1] // HOUR)

    def answer(self, label: str, hour: int) -> float:
        row = self.answers.get(hour)
        return float("nan") if row is None else float(row[self.labels.index(label)])


def prepare(dataset: Dataset, cache, labels: Sequence[str], window_s: float,
            dose_span_s: float, sample_weight: Optional[np.ndarray] = None) -> Dict[str, Recording]:
    """Every participant's Recording.

    Args:
        dataset: Checked for windows, events and patients (core/model.py does that).
        cache: harnet.FeatureCache (anything with get(participant, windows)).
        labels: The label names to keep answers for, such as the network's labels.
        window_s: The window length, from the window spec.
        dose_span_s: How long a dose stays marked after its time (tokens.event_fields).
        sample_weight: Optional weight per hour of dataset, in its hours order (core/model.py checks them).
    """
    labels = tuple(labels)
    events = dataset.events.groupby(PARTICIPANT) if len(dataset.events) else None
    patients = dataset.patients.set_index(PARTICIPANT)
    answer_table = dataset.labels.reindex(columns=list(labels)).astype("float32")
    answer_table[HOUR_INDEX] = dataset.hours[HOUR_INDEX].to_numpy()
    answer_table[PARTICIPANT] = dataset.hours[PARTICIPANT].to_numpy()
    if sample_weight is not None:
        answer_table["_weight"] = np.asarray(sample_weight, dtype=np.float64)
    out = {}
    for pid in dataset.participants:
        windows = dataset.windows[pid]
        worn = np.asarray(windows.worn, dtype=bool)
        time_s = np.asarray(windows.time_s, dtype=np.float64)[worn]
        features = cache.get(pid, windows)
        if len(features) != len(time_s):
            raise ModelError(f"{pid}: {len(features)} cached feature rows for {len(time_s)} worn windows")
        mine = events.get_group(pid) if events is not None and pid in events.groups else dataset.events.iloc[:0]
        fields = event_fields(time_s, mine, window_s, dose_span_s)
        if fields.dropped:
            logger.info("%s: %d event(s) after the last worn window, not placed", pid, fields.dropped,
                        extra={"event": "events_dropped", "participant": pid, "dropped": fields.dropped})
        row = patients.loc[pid]
        rows = answer_table[answer_table[PARTICIPANT] == pid]
        values = rows[list(labels)].to_numpy(np.float32)
        keep = ~np.isnan(values).all(axis=1)
        answers = {int(h): v for h, v in zip(rows[HOUR_INDEX].to_numpy()[keep], values[keep])}
        weights = None
        if sample_weight is not None:
            weights = {int(h): float(w) for h, w in zip(rows[HOUR_INDEX].to_numpy()[keep],
                                                        rows["_weight"].to_numpy()[keep])}
        out[pid] = Recording(pid, time_s, features, fields.fields,
                             patient_fields(row["more_affected_side"], row["dbs"]), answers, labels,
                             weights)
    return out


# ---- which chunks ------------------------------------------------------------------------------

def tile_chunks(recordings: Dict[str, Recording], chunk_h: int, rng: np.random.Generator,
                answered_only: bool = True) -> List[Tuple[str, int]]:
    """One epoch's chunks: [(participant, start hour)], back to back over each recording.

    Each recording is tiled from a random offset, so the boundaries move every epoch. With
    answered_only, a chunk with no answered hour is left out (nothing to grade in the diary step).
    """
    _check_hours("chunk_h", chunk_h)
    out = []
    for pid, rec in recordings.items():
        if not len(rec.time_s):
            continue
        first, last = rec.hours()
        start = first - int(rng.integers(0, chunk_h))
        while start <= last:
            if not answered_only or any(h in rec.answers for h in range(start, start + chunk_h)):
                out.append((pid, start))
            start += chunk_h
    return out


def batches(chunks: Sequence, batch_size: int, rng: np.random.Generator) -> List[list]:
    """The chunks shuffled and cut into batches of batch_size (the last one may be smaller)."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ConfigError(f"batch_size must be a positive integer, not {batch_size!r}")
    order = rng.permutation(len(chunks))
    return [[chunks[i] for i in order[k:k + batch_size]] for k in range(0, len(order), batch_size)]


def answered_hours(rec: Recording) -> List[int]:
    """The recording's answered hours, in order: validation scores each one on its own."""
    return sorted(rec.answers)


# ---- one example -------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Example:
    """One chunk of a recording, ready to collate.

    Attributes:
        participant: Its id.
        time_s: (n_all,) the windows' times: the n_input the model reads, then any future.
        features: (n_all, 1024) their harnet10 features.
        events: (n_input,) x 8 the input windows' event fields, with any simulated check-ins.
        patient: (2,) the patient fields.
        n_input: How many of the windows the model reads.
        hours: (n_hours,) the graded diary hours.
        targets: (n_hours, n_labels) their answers, NaN for none.
        check_ins: How many check-ins were simulated.
        weights: (n_hours,) the graded hours' weights (Recording.weights), or None for 1 each.
    """

    participant: str
    time_s: np.ndarray
    features: np.ndarray
    events: np.ndarray
    patient: np.ndarray
    n_input: int
    hours: np.ndarray
    targets: np.ndarray
    check_ins: int
    weights: Optional[np.ndarray] = None

    @property
    def has_graded_windows(self) -> bool:
        """True if any window the model reads is in a graded hour."""
        return bool(np.isin(self.time_s[:self.n_input] // HOUR, self.hours).any())


def make_example(rec: Recording, start_hour: int, hours: int, past_h: float, future_h: float = 0.0,
                 check_ins_per_day: float = 0.0, rng: Optional[np.random.Generator] = None) -> Example:
    """The chunk of `hours` diary hours from start_hour, with past_h hours of windows before it.

    Args:
        future_h: Hours of windows after the chunk to keep for pretraining's targets (0: none).
        check_ins_per_day: Simulated check-ins (0: none, as in validation and test).
        rng: For the check-ins; needed if there are any.
    """
    _check_hours("hours", hours)
    for name, value in (("past_h", past_h), ("future_h", future_h), ("check_ins_per_day", check_ins_per_day)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < float("inf"):
            raise ConfigError(f"{name} must be 0 or more, not {value!r}")
    begin, end = (start_hour - past_h) * HOUR, (start_hour + hours) * HOUR
    a, b, c = np.searchsorted(rec.time_s, [begin, end, end + future_h * HOUR])
    events = rec.events[a:b]
    n_check_ins = 0
    if check_ins_per_day:
        if rng is None:
            raise ConfigError("simulated check-ins need an rng")
        at, kinesia, tremor = _simulate_check_ins(rec, rec.time_s[a:b], check_ins_per_day, rng)
        if len(at):
            events = mark_check_ins(events, at, kinesia, tremor)
            n_check_ins = len(at)
    graded = np.arange(start_hour, start_hour + hours)
    targets = np.full((hours, len(rec.labels)), np.nan, dtype=np.float32)
    for i, h in enumerate(graded):
        if int(h) in rec.answers:
            targets[i] = rec.answers[int(h)]
    weights = (None if rec.weights is None else
               np.array([rec.weights.get(int(h), 1.0) for h in graded], dtype=np.float32))
    return Example(rec.participant, rec.time_s[a:c], np.asarray(rec.features[a:c], dtype=np.float32),
                   np.ascontiguousarray(events, dtype=np.float32), rec.patient, int(b - a), graded, targets,
                   n_check_ins, weights)


def _simulate_check_ins(rec, time_s, per_day, rng):
    """Check-ins on the answered waking hours among these windows: (token index, kinesia, tremor)."""
    if "state_now" not in rec.labels:
        return [], [], []
    state = rec.labels.index("state_now")
    tremor = rec.labels.index("tremor_now") if "tremor_now" in rec.labels else None
    hour_of = (time_s // HOUR).astype(np.int64)
    at, kinesia, tremors = [], [], []
    for h in np.unique(hour_of):
        row = rec.answers.get(int(h))
        if row is None or np.isnan(row[state]):          # unanswered, or asleep: no waking state
            continue
        if rng.random() < per_day / WAKING_HOURS_PER_DAY:
            in_hour = np.flatnonzero(hour_of == h)
            at.append(int(rng.choice(in_hour)))
            kinesia.append(float(row[state]) - 3)
            tremors.append(float(row[tremor]) if tremor is not None else float("nan"))
    return at, kinesia, tremors


# ---- a batch -----------------------------------------------------------------------------------

class Collated(NamedTuple):
    batch: Batch                                    # what the network reads (network.Batch)
    graded: torch.Tensor                            # (b, t) bool: input windows in a graded hour
    future: Optional[Tuple[torch.Tensor, ...]]      # (raw features, time_s, valid) for pretraining


def collate(examples: Sequence[Example], labels: Sequence[str], future: bool = False,
            pin: bool = False) -> Collated:
    """Examples padded into one batch (padding after each row's real windows).

    Args:
        labels: The label names, in the order the examples' targets columns are.
        future: Also collate each example's whole windows, future included, for pretraining.
        pin: Pin the tensors' memory, for a faster non-blocking copy to the GPU.
    """
    if not examples:
        raise ModelError("collate needs at least one example")
    n_hours = {len(e.hours) for e in examples}
    if len(n_hours) != 1:
        raise ModelError(f"every example in a batch needs the same number of graded hours, not {sorted(n_hours)}")
    b, t = len(examples), max(max(e.n_input for e in examples), 1)
    feature_dim = examples[0].features.shape[1]
    movement = np.zeros((b, t, feature_dim), dtype=np.float32)
    time_s = np.zeros((b, t), dtype=np.float64)
    valid = np.zeros((b, t), dtype=bool)
    events = np.zeros((b, t, examples[0].events.shape[1]), dtype=np.float32)
    for i, e in enumerate(examples):
        n = e.n_input
        movement[i, :n], time_s[i, :n], valid[i, :n], events[i, :n] = e.features[:n], e.time_s[:n], True, e.events
    hours = np.stack([e.hours for e in examples]).astype(np.int64)
    hour_of_window = np.where(valid, time_s // HOUR, -1).astype(np.int64)
    graded = valid & (hour_of_window[:, :, None] == hours[:, None, :]).any(-1)
    targets = np.stack([e.targets for e in examples])                          # (b, n_hours, n_labels)
    tensor = (lambda x: torch.from_numpy(x).pin_memory()) if pin else torch.from_numpy
    batch = Batch(movement=tensor(movement), time_s=tensor(time_s), valid=tensor(valid),
                  patient=tensor(np.stack([e.patient for e in examples])[:, None, :]), events=tensor(events),
                  hour_of_window=tensor(hour_of_window), hours=tensor(hours),
                  targets={label: tensor(np.ascontiguousarray(targets[..., j])) for j, label in enumerate(labels)},
                  sample_weights=None if all(e.weights is None for e in examples) else tensor(np.stack([
                      e.weights if e.weights is not None else np.ones(len(e.hours), np.float32) for e in examples])))
    ahead = None
    if future:
        tf = max(max(len(e.time_s) for e in examples), 1)
        f_features = np.zeros((b, tf, feature_dim), dtype=np.float32)
        f_time = np.zeros((b, tf), dtype=np.float64)
        f_valid = np.zeros((b, tf), dtype=bool)
        for i, e in enumerate(examples):
            n = len(e.time_s)
            f_features[i, :n], f_time[i, :n], f_valid[i, :n] = e.features, e.time_s, True
        ahead = (tensor(f_features), tensor(f_time), tensor(f_valid))
    return Collated(batch, tensor(graded), ahead)


def _check_hours(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{name} must be a positive whole number of hours, not {value!r}")


__all__ = ["Collated", "Example", "Recording", "answered_hours", "batches", "collate", "make_example",
           "prepare", "tile_chunks"]
