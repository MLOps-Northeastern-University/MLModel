"""Tokens: one per worn window, as the decoder sees it.

A token is three parts, each d_model numbers, added together:

    movement   LayerNorm(Linear(movement embedding))    what the movement encoder read (movement.py)
    patient    LayerNorm(Linear(patient fields))        watch on the more affected side; DBS
    events     Linear(event fields), no bias            a dose or check-in at this window, else exactly 0

Each part is brought to the same size, and the two that are always present are normalised, so
neither outweighs the other at the start: the 256 movement numbers don't drown the 2 patient
ones. Each part's Linear learns which directions of the token it writes to, so later layers can
still read each part back from the sum, as transformers add word and position embeddings
(Vaswani et al. 2017). The event part isn't normalised, so "nothing happened" stays exactly 0.

Event fields, 0 except where something happened:

    dose           1 on a window a dose marks (see dose span below)
    ledd           the dose's LEDD / ledd_scale; doses marking the same window add up
    on_time, late, none_logged    the dose's source, one-hot; the least trustworthy if several
    check_in       1 on the window where the patient answered a check-in
    kinesia        the answer's kinesia score / 3 (-1 to +1)
    tremor         the answer's tremor score / 2 (0 to 1)

The dose and check_in flags tell "nothing happened" apart from "a check-in answering 0".

Where an event goes. Only worn windows are tokens, so an event goes on the window that holds its
time, or, if the watch wasn't worn then, on the next worn window: live, the model learns of it
at the first window after it. An event after the last token has nowhere to go and is counted.

Once per recording, then sliced. ``event_fields`` is meant for a participant's whole recording:
all their worn windows and all their events. Each training or scoring chunk then takes its rows
of the result, and so gets exactly what it would have had live. Working on a chunk alone can't:
it can't tell the chunk's start from the watch being off, so a dose from hours before would
land on its first window, or, with earlier events left out, a dose 10 minutes before its start
would lose the 20 minutes of span it still has. It's also faster: once per participant, not
once per chunk.

Check-ins. Real ones come from the app, as events. COPS has none, so training simulates them:
``mark_check_ins`` adds them to any of a chunk's rows (the past it reads as well as the hours it
is graded on), on a copy, so the recording's fields stay as they were. How many, where, and how
validation scores with and without them are the training step's settings, not this file's.

Dose span. A dose marks every window from its own up to ``dose_span_s`` after its time (30 min
in the final recipe: 180 windows), never before it, since live a dose is only known once logged.
A 10-s mark was too faint for attention to find. Doses whose spans overlap add up their LEDD.

Speed. ``event_fields`` runs on the CPU and only loops over events, never windows: each event
is placed by binary search, and each dose marks its span with one slice. Over all 15 local
participants (588,233 windows) it takes about 0.01 s. The patient part is the same for every
token of a participant, so
``TokenBuilder`` accepts patient fields of shape (batch, 1, 2) and broadcasts them, rather than
needing a copy per token.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.core.data import DOSE_SOURCES, EVENT_KINDS
from scripts.core.errors import ConfigError, ModelError, SchemaError
from scripts.core.targets import KINESIA_SCORES, TREMOR_SCORES
from scripts.models.motor_causal_transformer.init import init_weights

PATIENT_FIELDS = ("more_affected_side", "dbs")
EVENT_FIELDS = ("dose", "ledd", "on_time", "late", "none_logged", "check_in", "kinesia", "tremor")
_F = {name: i for i, name in enumerate(EVENT_FIELDS)}
# Dose sources from most to least trustworthy, as the dataset contract orders them.
if tuple(DOSE_SOURCES) != EVENT_FIELDS[2:5]:
    raise ImportError(f"the source fields {EVENT_FIELDS[2:5]} must follow core.data.DOSE_SOURCES {DOSE_SOURCES}")
_SOURCE_RANK = {source: rank for rank, source in enumerate(DOSE_SOURCES)}
KINESIA_SCALE, TREMOR_SCALE = 3.0, 2.0   # the largest scores, so answers land in -1..1 and 0..1


@dataclass(frozen=True)
class EventFields:
    """The event fields of a run of tokens.

    Attributes:
        fields: (n_tokens, len(EVENT_FIELDS)) float32.
        dropped: Events after the last token, with no token to go on.
    """

    fields: np.ndarray
    dropped: int


def event_fields(token_time_s: np.ndarray, events: pd.DataFrame, window_s: float,
                 dose_span_s: float = 0.0, ledd_scale: float = 100.0) -> EventFields:
    """The event fields of each token of one participant's whole recording.

    Args:
        token_time_s: (n,) the start time in seconds of each of the participant's worn windows,
            increasing: the whole recording, so that chunks can be sliced from the result.
        events: All the participant's events: time_s, kind, ledd, source, kinesia_score,
            tremor_score, as the dataset contract checks them (core/data.py).
        window_s: The window length in seconds (from the window spec).
        dose_span_s: How long after its time a dose stays marked (0: its own window only).
        ledd_scale: LEDD is divided by this, to be on the scale of the other fields.

    Raises:
        ModelError: token times that aren't increasing, or settings out of range.
        SchemaError: an event of an unknown kind, or a dose of an unknown source.
    """
    if not (window_s > 0 and math.isfinite(window_s)):
        raise ModelError(f"window_s must be a positive number of seconds, not {window_s!r}")
    if not (dose_span_s >= 0 and math.isfinite(dose_span_s)):
        raise ModelError(f"dose_span_s must be 0 or more seconds, not {dose_span_s!r}")
    if not (ledd_scale > 0 and math.isfinite(ledd_scale)):
        raise ModelError(f"ledd_scale must be positive, not {ledd_scale!r}")
    t = np.asarray(token_time_s, dtype=np.float64)
    if t.ndim != 1 or (len(t) > 1 and np.any(np.diff(t) <= 0)):
        raise ModelError("token_time_s must be one increasing time per token")
    n = len(t)
    fields = np.zeros((n, len(EVENT_FIELDS)), dtype=np.float32)
    if len(events) == 0 or n == 0:
        return EventFields(fields, len(events))

    events = events.sort_values("time_s", kind="stable")   # in time order, so a later check-in wins a window
    times = events["time_s"].to_numpy(dtype=np.float64)
    # The token whose window holds the event, else the next token.
    holder = np.searchsorted(t, times, side="right") - 1
    inside = (holder >= 0) & (times < t[np.maximum(holder, 0)] + window_s)
    index = np.where(inside, holder, holder + 1)
    placed = index < n
    kind = events["kind"].to_numpy()
    unknown = ~np.isin(kind, EVENT_KINDS)
    if unknown.any():
        raise SchemaError(f"events.kind must be one of {EVENT_KINDS}, not {kind[unknown][0]!r}")

    doses = placed & (kind == "dose")
    if doses.any():
        first = index[doses]
        # The windows starting before time + span, and always at least its own.
        last = np.searchsorted(t, times[doses] + dose_span_s, side="left") if dose_span_s else first + 1
        last = np.maximum(last, first + 1)
        ledd = events["ledd"].to_numpy(dtype=np.float64)[doses] / ledd_scale
        sources = events["source"].to_numpy()[doses]
        unknown = ~np.isin(sources, DOSE_SOURCES)
        if unknown.any():
            raise SchemaError(f"events.source of a dose must be one of {DOSE_SOURCES}, not {sources[unknown][0]!r}")
        rank = np.array([_SOURCE_RANK[s] for s in sources])
        source_rank = np.full(n, -1)
        for a, b, dose_ledd, r in zip(first, last, ledd, rank):   # one slice per dose, never a loop over windows
            fields[a:b, _F["dose"]] = 1.0
            fields[a:b, _F["ledd"]] += dose_ledd
            np.maximum(source_rank[a:b], r, out=source_rank[a:b])
        marked = source_rank >= 0
        fields[marked, _F["on_time"] + source_rank[marked]] = 1.0   # the least trustworthy source, one-hot

    check_ins = placed & (kind == "check_in")
    if check_ins.any():
        _mark(fields, index[check_ins], events["kinesia_score"].to_numpy(dtype=np.float64)[check_ins],
              events["tremor_score"].to_numpy(dtype=np.float64)[check_ins])
    return EventFields(fields, int((~placed).sum()))


def mark_check_ins(fields: np.ndarray, at, kinesia_score, tremor_score) -> np.ndarray:
    """A copy of a chunk's event fields with check-ins added, such as simulated ones in training.

    Args:
        fields: (n, len(EVENT_FIELDS)) a chunk's rows of event_fields. Not changed: it is
            usually a view of the whole recording's fields.
        at: (k,) the token each check-in is answered on.
        kinesia_score, tremor_score: (k,) the answers; NaN where that question wasn't answered.

    Raises:
        ModelError: a token out of range, an answer that isn't a score, or a check-in with
            neither answer.
    """
    at = np.asarray(at, dtype=np.int64).reshape(-1)
    kinesia, tremor = (np.asarray(v, dtype=np.float64).reshape(-1) for v in (kinesia_score, tremor_score))
    if not len(at) == len(kinesia) == len(tremor):
        raise ModelError("mark_check_ins needs one token, kinesia and tremor answer per check-in")
    if len(at) and (at.min() < 0 or at.max() >= len(fields)):
        raise ModelError(f"check-ins must be on tokens 0 to {len(fields) - 1}")
    for name, values, allowed in (("kinesia", kinesia, KINESIA_SCORES), ("tremor", tremor, TREMOR_SCORES)):
        answered = values[~np.isnan(values)]
        if not np.isin(answered, allowed).all():
            raise ModelError(f"a {name} answer must be one of {allowed} or NaN")
    if (np.isnan(kinesia) & np.isnan(tremor)).any():
        raise ModelError("a check-in needs at least one answer")
    out = fields.copy()
    _mark(out, at, kinesia, tremor)
    return out


def _mark(fields, at, kinesia, tremor):
    """Mark check-ins in place: the flag, and each answer scaled. Two on one token: the later answer wins."""
    fields[at, _F["check_in"]] = 1.0
    for score, field, scale in ((kinesia, "kinesia", KINESIA_SCALE), (tremor, "tremor", TREMOR_SCALE)):
        answered = ~np.isnan(score)
        fields[at[answered], _F[field]] = score[answered] / scale


def patient_fields(more_affected_side, dbs) -> np.ndarray:
    """One participant's patient fields: (len(PATIENT_FIELDS),) float32, each 1 or 0.

    Raises:
        SchemaError: a field that isn't True or False (a missing value is not "no").
    """
    values = []
    for name, value in zip(PATIENT_FIELDS, (more_affected_side, dbs)):
        if not isinstance(value, (bool, np.bool_)):
            raise SchemaError(f"patients.{name} must be True or False, not {value!r}")
        values.append(float(value))
    return np.array(values, dtype=np.float32)


class TokenBuilder(nn.Module):
    """The three parts of a token, each d_model numbers, added: movement + patient + events.

    Args:
        movement_dim: Numbers per window from the movement encoder (its output_dim).
        d_model: Numbers per token.
        dropout: Dropout on the finished token (0.3 in the final recipe).
    """

    def __init__(self, movement_dim: int = 256, d_model: int = 64, dropout: float = 0.3):
        super().__init__()
        for name, value in (("movement_dim", movement_dim), ("d_model", d_model)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError(f"tokens {name} must be a positive integer, not {value!r}")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ConfigError(f"tokens dropout must be at least 0 and below 1, not {dropout!r}")
        self.movement_dim, self.d_model = movement_dim, d_model
        self.movement = nn.Sequential(nn.Linear(movement_dim, d_model), nn.LayerNorm(d_model))
        self.patient = nn.Sequential(nn.Linear(len(PATIENT_FIELDS), d_model), nn.LayerNorm(d_model))
        self.events = nn.Linear(len(EVENT_FIELDS), d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.apply(init_weights)
        # The event part has no LayerNorm (so "nothing happened" stays 0), so with plain Xavier a
        # dose's part would be about 3x shorter than the normalised parts. Scaling Xavier so each
        # weight has SD 1/sqrt(3) gives a dose (3 fields near 1: dose, ledd, source) a part of
        # length about sqrt(d_model), the same as a LayerNorm'd part, so it's visible from the start.
        xavier_sd = math.sqrt(2 / (len(EVENT_FIELDS) + d_model))
        nn.init.xavier_uniform_(self.events.weight, gain=1 / (math.sqrt(3) * xavier_sd))

    def forward(self, movement: torch.Tensor, patient: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        """Build the tokens.

        Args:
            movement: (batch, seq, movement_dim) from the movement encoder.
            patient: (batch, 1, 2) per participant, or (batch, seq, 2) per token.
            events: (batch, seq, 8) from event_fields.

        Returns:
            (batch, seq, d_model).
        """
        if movement.dim() != 3 or movement.shape[-1] != self.movement_dim:
            raise ModelError(f"movement must be (batch, seq, {self.movement_dim}), not {tuple(movement.shape)}")
        lead: Tuple[int, int] = movement.shape[:2]
        if events.shape != (*lead, len(EVENT_FIELDS)):
            raise ModelError(f"events must be {(*lead, len(EVENT_FIELDS))}, not {tuple(events.shape)}")
        if patient.dim() != 3 or patient.shape[0] != lead[0] or patient.shape[1] not in (1, lead[1]) \
                or patient.shape[2] != len(PATIENT_FIELDS):
            raise ModelError(f"patient must be ({lead[0]}, 1 or {lead[1]}, {len(PATIENT_FIELDS)}), "
                             f"not {tuple(patient.shape)}")
        return self.dropout(self.movement(movement) + self.patient(patient) + self.events(events))
