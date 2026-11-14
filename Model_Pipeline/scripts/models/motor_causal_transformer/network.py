"""The motor causal transformer's network: every part, wired together.

    movement input  -> movement encoder -> one embedding per window      (movement.py)
    + patient, events                   -> one token per window           (tokens.py)
    tokens                              -> decoder: each window's output  (decoder.py)
    decoder output                      -> heads: each window's logits    (heads.py)
    window probabilities                -> pooling: each graded hour      (pooling.py)
    hours and windows vs the diary      -> loss                           (loss.py)

The parts are named in PARTS. The names are the state dict's prefixes, which freezing and loading
pretrained weights work by:

    pretraining   the whole network learns from movement and check-ins (pretraining.py)
    diary step    load_parts(pretrained, ("movement", "tokens", "decoder")), then
                  freeze_all_but(("heads", "pooling")): only those learn

Freezing stops a part's weights changing and keeps the part in eval mode, so its dropout is off
even while the rest trains. Dropout in a part that doesn't learn regularises nothing. It would
only train the heads on noisier features than they are judged on. The prototype left it on;
``keep_dropout=True`` does the same, to compare.

d_model and dropout are shared by the token builder, decoder and heads. Every other setting goes
to its own part, so config mirrors the parts:

    MotorNetwork(movement="harnet10", movement_params={...}, d_model=64, dropout=0.3,
                 decoder={...}, heads={...}, pooling={...}, loss={...})

``settings`` gives those arguments back, so a saved model rebuilds exactly the network it was
trained with.
"""

from dataclasses import dataclass, fields
from typing import Dict, Iterable, Mapping, NamedTuple, Optional, Tuple

import torch
from torch import nn

from scripts.core.errors import ConfigError, ModelError
from scripts.models.motor_causal_transformer.decoder import Decoder
from scripts.models.motor_causal_transformer.heads import Heads, probabilities
from scripts.models.motor_causal_transformer.loss import LossResult, MILLoss
from scripts.models.motor_causal_transformer.movement import build_movement_encoder
from scripts.models.motor_causal_transformer.pooling import GatedPooling, Pooled, hour_members
from scripts.models.motor_causal_transformer.tokens import EVENT_FIELDS, PATIENT_FIELDS, TokenBuilder

PARTS = ("movement", "tokens", "decoder", "heads", "pooling")


@dataclass(frozen=True, eq=False)   # eq=False: == on tensors isn't a yes or no
class Batch:
    """One batch of sequences of worn windows, padded to the same length.

    Attributes:
        movement: (batch, seq, *input_shape) the movement encoder's input, such as harnet10's
            cached features (batch, seq, 1024).
        time_s: (batch, seq) each window's time in seconds, increasing along each sequence.
        valid: (batch, seq) bool, False for padding, which comes after the real windows.
        patient: (batch, 1, 2) PATIENT_FIELDS per sequence.
        events: (batch, seq, 8) EVENT_FIELDS per window.
        hour_of_window: (batch, seq) integers, each window's diary hour. Needed to pool.
        hours: (batch, n_hours) integers, the hours to pool and grade. Needed to pool.
        targets: {label: (batch, n_hours)} class indices as floats, NaN for no answer. Needed for the loss.
        sample_weights: (batch, n_hours) each graded hour's weight in the loss, or None for 1 each.
    """

    movement: torch.Tensor
    time_s: torch.Tensor
    valid: torch.Tensor
    patient: torch.Tensor
    events: torch.Tensor
    hour_of_window: Optional[torch.Tensor] = None
    hours: Optional[torch.Tensor] = None
    targets: Optional[Mapping[str, torch.Tensor]] = None
    sample_weights: Optional[torch.Tensor] = None

    def __post_init__(self):
        b_t = tuple(self.time_s.shape)
        if len(b_t) != 2:
            raise ModelError(f"time_s must be (batch, seq), not {b_t}")
        if tuple(self.valid.shape) != b_t or self.valid.dtype != torch.bool:
            raise ModelError(f"valid must be bool {b_t}, not {self.valid.dtype} {tuple(self.valid.shape)}")
        if tuple(self.movement.shape[:2]) != b_t:
            raise ModelError(f"movement must start {b_t}, not {tuple(self.movement.shape)}")
        if tuple(self.events.shape) != (*b_t, len(EVENT_FIELDS)):
            raise ModelError(f"events must be {(*b_t, len(EVENT_FIELDS))}, not {tuple(self.events.shape)}")
        if tuple(self.patient.shape) != (b_t[0], 1, len(PATIENT_FIELDS)):
            raise ModelError(f"patient must be {(b_t[0], 1, len(PATIENT_FIELDS))}, not {tuple(self.patient.shape)}")
        if (self.hour_of_window is None) != (self.hours is None):
            raise ModelError("hour_of_window and hours go together: give both or neither")
        if self.targets is not None and self.hours is None:
            raise ModelError("targets need hour_of_window and hours, to know which windows they grade")
        if self.sample_weights is not None and (self.hours is None or self.sample_weights.shape != self.hours.shape):
            raise ModelError(f"sample_weights must be (batch, n_hours) like hours, not "
                             f"{tuple(self.sample_weights.shape)}")
        # The rest of their checks are in hour_members and the loss.

    def to(self, device, non_blocking: bool = False) -> "Batch":
        """The same batch on another device. Floats stay as they are; mixed precision is autocast's job."""
        def move(x):
            if x is None:
                return None
            if isinstance(x, Mapping):
                return {k: v.to(device, non_blocking=non_blocking) for k, v in x.items()}
            return x.to(device, non_blocking=non_blocking)
        return Batch(**{f.name: move(getattr(self, f.name)) for f in fields(self)})


class Output(NamedTuple):
    logits: Dict[str, torch.Tensor]            # {label: (batch, seq, classes)} every window's
    pooled: Optional[Dict[str, Pooled]]        # {label: Pooled} every graded hour's, if the batch has hours
    has_windows: Optional[torch.Tensor]        # (batch, n_hours) hours with worn windows, if it has hours
    loss: Optional[LossResult]                 # if the batch has targets


class MotorNetwork(nn.Module):
    """The network: movement encoder, token builder, decoder, heads, pooling and loss.

    Args:
        movement: The movement encoder's name (movement.py's ENCODERS).
        movement_params: Its own settings.
        d_model: Numbers per token, shared by the tokens, decoder, heads and pooling (64).
        dropout: Dropout of the tokens, decoder and heads (0.3).
        decoder, heads, pooling, loss: Each part's other settings, passed on as they are.

    Raises:
        ConfigError: a setting a part doesn't take, or a value it refuses.
    """

    def __init__(self, movement: str = "harnet10", movement_params: Optional[Mapping] = None,
                 d_model: int = 64, dropout: float = 0.3, decoder: Optional[Mapping] = None,
                 heads: Optional[Mapping] = None, pooling: Optional[Mapping] = None,
                 loss: Optional[Mapping] = None):
        super().__init__()
        self.settings = {"movement": movement, "movement_params": dict(movement_params or {}),
                         "d_model": d_model, "dropout": dropout, "decoder": dict(decoder or {}),
                         "heads": dict(heads or {}), "pooling": dict(pooling or {}), "loss": dict(loss or {})}
        s = self.settings
        self.movement = build_movement_encoder(movement, s["movement_params"])
        self.tokens = TokenBuilder(self.movement.output_dim, d_model, dropout)
        self.decoder = _build("decoder", Decoder, s["decoder"], d_model=d_model, dropout=dropout)
        self.heads = _build("heads", Heads, s["heads"], d_model=d_model, dropout=dropout)
        self.pooling = _build("pooling", GatedPooling, s["pooling"], labels=self.heads.labels, d_model=d_model)
        self.loss = _build("loss", MILLoss, s["loss"])
        self._frozen: Tuple[str, ...] = ()   # parts kept in eval mode (freeze_all_but)

    def train(self, mode: bool = True) -> "MotorNetwork":
        """As nn.Module.train, except that frozen parts stay in eval mode."""
        super().train(mode)
        for part in self._frozen:
            getattr(self, part).eval()
        return self

    @property
    def labels(self) -> Tuple[str, ...]:
        """Every output's label, in the heads' order."""
        return self.heads.labels

    def encode(self, batch: Batch) -> torch.Tensor:
        """Every window's decoder output, (batch, seq, d_model). Pretraining reads this."""
        b, t = batch.time_s.shape
        embedding = self.movement(batch.movement.flatten(0, 1)).unflatten(0, (b, t))
        tokens = self.tokens(embedding, batch.patient, batch.events)
        return self.decoder(tokens, batch.time_s, batch.valid)

    def forward(self, batch: Batch) -> Output:
        """Every window's logits; with hours, every graded hour's prediction; with targets, the loss.

        Pooling and the loss run in float32 by themselves, so under bf16 autocast only the
        encoder, tokens, decoder and heads run in bf16.
        """
        h = self.encode(batch)
        logits = self.heads(h)
        if batch.hours is None:
            return Output(logits, None, None, None)
        members = hour_members(batch.hour_of_window, batch.valid, batch.hours)
        pooled, has_windows = self.pooling(h, probabilities(logits), members)
        loss = None
        if batch.targets is not None:
            loss = self.loss(pooled, logits, members, batch.targets, has_windows, batch.sample_weights)
        return Output(logits, pooled, has_windows, loss)

    # ---- training helpers ----------------------------------------------------------------------

    def freeze_all_but(self, parts: Iterable[str], keep_dropout: bool = False) -> Tuple[str, ...]:
        """Only these parts learn; the rest are frozen and kept in eval mode. All of PARTS unfreezes everything.

        Args:
            parts: The parts that learn, from PARTS.
            keep_dropout: Leave the frozen parts' dropout on while training, as the prototype did.

        Returns:
            The names of the weights that learn, for the log.
        """
        parts = _check_parts(parts)
        for name, p in self.named_parameters():
            p.requires_grad_(name.split(".", 1)[0] in parts)
        self._frozen = () if keep_dropout else tuple(part for part in PARTS if part not in parts)
        self.train(self.training)
        return tuple(name for name, p in self.named_parameters() if p.requires_grad)

    def load_parts(self, state: Mapping[str, torch.Tensor], parts: Iterable[str]) -> None:
        """Copy these parts' weights (and buffers, such as the movement standardisation) from a state dict.

        Every weight of those parts must be in state and fit, so a pretrained network that
        doesn't match fails here, not quietly halfway through training. The other parts stay as
        they are.
        """
        parts = _check_parts(parts)
        own = {k: v for k, v in self.state_dict().items() if k.split(".", 1)[0] in parts}
        missing = sorted(set(own) - set(state))
        if missing:
            raise ModelError(f"the pretrained weights are missing {len(missing)} of {sorted(parts)}'s, "
                             f"such as {missing[:3]}")
        wrong = [k for k in own if tuple(state[k].shape) != tuple(own[k].shape)]
        if wrong:
            raise ModelError(f"pretrained weights of the wrong shape, such as {wrong[0]}: "
                             f"{tuple(state[wrong[0]].shape)}, not {tuple(own[wrong[0]].shape)}")
        self.load_state_dict({**self.state_dict(), **{k: state[k] for k in own}})

    def parameter_groups(self, weight_decay: float):
        """AdamW groups for the weights that learn: decay on weight matrices only.

        Biases, LayerNorm gains and the pooling temperatures aren't decayed. Pulling them towards
        0 only shrinks a scale or shift; it doesn't regularise anything.
        """
        if isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float)) or weight_decay < 0:
            raise ConfigError(f"weight_decay must be a number of 0 or more, not {weight_decay!r}")
        learn = [p for p in self.parameters() if p.requires_grad]
        if not learn:
            raise ModelError("every weight is frozen: nothing to train")
        return [{"params": [p for p in learn if p.ndim >= 2], "weight_decay": float(weight_decay)},
                {"params": [p for p in learn if p.ndim < 2], "weight_decay": 0.0}]

    def size(self) -> Dict[str, int]:
        """Weights per part, and how many of them learn, for the log."""
        out = {part: sum(p.numel() for p in getattr(self, part).parameters()) for part in PARTS}
        out["learning"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return out


def _build(part, cls, params, **shared):
    """A part from its own settings plus the shared ones, with config errors in plain words."""
    clash = sorted(set(params) & set(shared))
    if clash:
        raise ConfigError(f"{part} settings can't set {clash}: they're shared and set once for the network")
    try:
        return cls(**shared, **params)
    except TypeError as e:
        raise ConfigError(f"{part} settings: {e}") from None


def _check_parts(parts):
    parts = frozenset([parts] if isinstance(parts, str) else parts)
    unknown = sorted(parts - set(PARTS))
    if unknown:
        raise ConfigError(f"unknown network parts {unknown}; known: {list(PARTS)}")
    return parts


__all__ = ["Batch", "MotorNetwork", "Output", "PARTS"]
