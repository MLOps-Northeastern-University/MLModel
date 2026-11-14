"""MIL pooling: an hour's window predictions into one prediction for that hour.

The diary is answered once an hour, but the model predicts every 10-s window. Multiple-instance
learning (MIL) treats each hour as a bag of windows with a single label. Pooling turns the bag's
window predictions into the hour's prediction, and the loss compares that with the diary. This
is gated attention pooling (Ilse, Tomczak and Welling 2018, "Attention-based Deep Multiple
Instance Learning", ICML), with two changes, both from the prototype:

    score    = w . (tanh(V h) * sigmoid(U h))    one score per window and output; h is the decoder's
    weights  = softmax(score / temperature)      over the hour's worn windows only
    hour     = sum(weights * window probabilities)

1. Probabilities are pooled, not embeddings. (The paper pools embeddings and then classifies.)
   Every window keeps its own prediction, which live use needs, and the hour's prediction is a
   weighted mix of them, so it always adds up to 1.
2. Each output (sleep, state and tremor, now and next hour) has its own learned temperature:
       large  every window weighs the same: the plain average
       small  a few windows decide
   So each output learns how to read its hour: a state that lasts the hour gets averaged, and
   tremor that comes and goes gets focused on. The temperature is learned as a log, so it stays
   positive.

V, U and w are shared by every output, except that w has one row per output. The parameter names
(V, U, score, log_temperature) are the prototype's, so its weights load unchanged.

The plain average of each hour's windows comes back too, so evaluation can compare it with the
attention pooling. The loss doesn't need it: its per-window term (loss.py) trains every window
directly, including windows the pooling ignores.

Unworn windows and padding get exactly zero weight. An hour with no worn windows gets no
prediction: its has_windows is False and its rows are zeros, never NaN.

Precision: pooling always runs in float32, even under autocast. Its sums and the loss's logs
need it, and it is a tiny share of the time. Speed: one gate pass and one masked softmax for
every output together, then one small matmul per output. The plain-average weights are worked
out once and shared. There is no GPU sync.
"""

import math
from typing import Dict, NamedTuple, Sequence, Tuple

import torch
from torch import nn

from scripts.core.errors import ConfigError, ModelError
from scripts.models.motor_causal_transformer.init import init_weights


class Pooled(NamedTuple):
    """One output's hours."""

    attention: torch.Tensor   # (batch, hours, classes) the hour's prediction
    mean: torch.Tensor        # (batch, hours, classes) the plain average of its windows
    weights: torch.Tensor     # (batch, hours, seq) each window's weight in each hour


def hour_members(hour_of_window: torch.Tensor, worn: torch.Tensor, hours: torch.Tensor) -> torch.Tensor:
    """Which worn windows belong to each hour asked for.

    Args:
        hour_of_window: (batch, seq) integers, each window's diary hour (any value for padding).
        worn: (batch, seq) bool, False for unworn windows and padding.
        hours: (batch, n_hours) integers, the hours to pool, such as the hours an example is graded on.

    Returns:
        (batch, n_hours, seq) bool.

    Raises:
        ModelError: shapes that don't match, or the wrong dtypes.
    """
    if hour_of_window.dim() != 2 or worn.shape != hour_of_window.shape:
        raise ModelError(f"hour_of_window and worn must both be (batch, seq), not "
                         f"{tuple(hour_of_window.shape)} and {tuple(worn.shape)}")
    if hours.dim() != 2 or hours.shape[0] != hour_of_window.shape[0]:
        raise ModelError(f"hours must be (batch, n_hours) for batch {hour_of_window.shape[0]}, "
                         f"not {tuple(hours.shape)}")
    if worn.dtype != torch.bool:
        raise ModelError(f"worn must be bool, not {worn.dtype}")
    if hour_of_window.is_floating_point() or hours.is_floating_point():
        raise ModelError("hour_of_window and hours must be integers")
    return (hour_of_window.unsqueeze(1) == hours.unsqueeze(2)) & worn.unsqueeze(1)


def _at_least_float32(x: torch.Tensor) -> torch.Tensor:
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


class GatedPooling(nn.Module):
    """Gated attention scores per window and output, a learned temperature per output, and the pooling.

    Args:
        labels: The outputs pooled, such as ("sleep_now", "state_now", ...), in the heads' order.
        d_model: Width of the decoder output it reads (64).
        gate_dim: Width of the gate (64).
        init_temperature: Every temperature's starting value (1.0).
    """

    def __init__(self, labels: Sequence[str], d_model: int = 64, gate_dim: int = 64,
                 init_temperature: float = 1.0):
        super().__init__()
        labels = tuple(labels)
        if not labels or len(set(labels)) != len(labels) or not all(isinstance(x, str) and x for x in labels):
            raise ConfigError(f"pooling labels must be distinct non-empty names, not {labels!r}")
        for name, value in (("d_model", d_model), ("gate_dim", gate_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError(f"pooling {name} must be a positive integer, not {value!r}")
        if (isinstance(init_temperature, bool) or not isinstance(init_temperature, (int, float))
                or not math.isfinite(init_temperature) or init_temperature <= 0):
            raise ConfigError(f"pooling init_temperature must be a positive number, not {init_temperature!r}")
        self.labels, self.d_model = labels, d_model
        self.V, self.U = nn.Linear(d_model, gate_dim), nn.Linear(d_model, gate_dim)
        self.score = nn.Linear(gate_dim, len(labels))   # w: one row per output
        self.apply(init_weights)
        self.log_temperature = nn.Parameter(torch.full((len(labels),), math.log(init_temperature)))

    @property
    def temperatures(self) -> Dict[str, float]:
        """{label: temperature}, for logging."""
        return dict(zip(self.labels, self.log_temperature.detach().exp().tolist()))

    def scores(self, h: torch.Tensor) -> torch.Tensor:
        """(batch, seq, d_model) -> (batch, seq, labels): each window's score for each output."""
        return self.score(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))

    def forward(self, h: torch.Tensor, window_probs: Dict[str, torch.Tensor],
                members: torch.Tensor) -> Tuple[Dict[str, Pooled], torch.Tensor]:
        """Pool every output's window probabilities into its hours.

        Args:
            h: (batch, seq, d_model) the decoder's output.
            window_probs: {label: (batch, seq, classes)} each window's probabilities, for every label.
            members: (batch, hours, seq) bool, from hour_members.

        Returns:
            ({label: Pooled}, has_windows (batch, hours) bool), all in float32 or wider.

        Raises:
            ModelError: missing or extra labels, or shapes that don't match.
        """
        self._check(h, window_probs, members)
        with torch.autocast(device_type=h.device.type, enabled=False):
            h = _at_least_float32(h)
            has_windows = members.any(dim=-1)                                   # (b, H)
            logits = (self.scores(h) / self.log_temperature.exp()).transpose(1, 2).unsqueeze(2)  # (b, L, 1, t)
            # An hour with no worn windows keeps its scores, so its softmax isn't NaN (all -inf),
            # and then gets zero weights from the multiply by members.
            keep = (members | ~has_windows.unsqueeze(-1)).unsqueeze(1)           # (b, 1, H, t)
            weights = torch.softmax(torch.where(keep, logits, -math.inf), dim=-1)
            weights = weights * members.unsqueeze(1)                            # (b, L, H, t)
            counts = members.sum(dim=-1, keepdim=True).clamp_min(1)
            mean_weights = members.to(weights.dtype) / counts                   # (b, H, t), shared
            out = {}
            for i, label in enumerate(self.labels):
                probs = _at_least_float32(window_probs[label])
                out[label] = Pooled(weights[:, i] @ probs, mean_weights @ probs, weights[:, i])
        return out, has_windows

    def _check(self, h, window_probs, members):
        if set(window_probs) != set(self.labels):
            raise ModelError(f"pooling needs window probabilities for {sorted(self.labels)}, "
                             f"not {sorted(window_probs)}")
        if h.dim() != 3 or h.shape[-1] != self.d_model:
            raise ModelError(f"pooling reads (batch, seq, {self.d_model}), not {tuple(h.shape)}")
        b, t = h.shape[:2]
        if members.dtype != torch.bool or members.dim() != 3 or members.shape[0] != b or members.shape[2] != t:
            raise ModelError(f"members must be bool (batch {b}, hours, seq {t}), not "
                             f"{members.dtype} {tuple(members.shape)}")
        for label, probs in window_probs.items():
            if probs.dim() != 3 or probs.shape[:2] != (b, t):
                raise ModelError(f"{label} window probabilities must be ({b}, {t}, classes), not "
                                 f"{tuple(probs.shape)}")


__all__ = ["GatedPooling", "Pooled", "hour_members"]
