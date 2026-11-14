"""The heads: every window's sleep, state and tremor, now and in the next hour.

Each horizon (now, next_hour) has its own head, and each head reads the decoder's output for a
window:

    Linear d_model -> hidden, GELU, Dropout      the head's trunk
    Linear hidden -> 12, split into              one output layer for all three targets
        sleep    2 classes   awake, asleep
        state    7 classes   kinesia -3 .. +3 of an awake window (P(state | awake))
        tremor   3 classes   tremor score 0, 1, 2

The targets, their classes and the horizons come from core/targets.py, so the labels are named
the same way everywhere: sleep_now, state_now, tremor_now, then the same for next_hour, in that
order (the order pooling uses).

Asleep is its own output, not an 8th state class: as one softmax, still OFF windows were filed
as asleep, taking probability from OFF exactly when the patient was OFF (core/targets.py). So
P(OFF) = P(awake) x P(OFF | awake), and the state is learned from waking hours only.

The next hour is a head of its own reading the same output, not a step-by-step rollout.

The outputs are logits, which the loss needs for numerical stability. probabilities() turns them
into probabilities, in float32 even under mixed precision.

Speed: each horizon's three output layers are a single Linear (12 outputs) whose result is split
into views, with no copy. That is 0.7 ms less per training step than three separate layers
(forward + backward, 4 x 3,240 windows, RTX 4060), with the same maths. The state dict therefore
has one out.weight per horizon. Rows 0-1 are sleep, 2-8 state and 9-11 tremor; the prototype
had separate out.sleep, out.state and out.tremor layers.

set_priors starts each output predicting the training hours' class mix, so the first steps
don't spend themselves learning the base rates.
"""

from typing import Dict, Mapping, Sequence

import torch
from torch import nn

from scripts.core.errors import ConfigError, ModelError
from scripts.core.targets import HORIZONS, TARGETS, label_name
from scripts.models.motor_causal_transformer.init import init_weights

# Classes per target, in the order each head's output layer gives them.
TARGET_CLASSES: Dict[str, int] = {name: len(target.classes) for name, target in TARGETS.items()}


class Head(nn.Module):
    """One horizon: a small hidden layer, then one output layer for every target."""

    def __init__(self, d_model: int, hidden: int, dropout: float):
        super().__init__()
        # hidden 0: the output layer reads the decoder's output directly (a linear head).
        self.trunk = (nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout))
                      if hidden else nn.Identity())
        self.out = nn.Linear(hidden or d_model, sum(TARGET_CLASSES.values()))

    def rows(self, target: str) -> slice:
        """The output layer's rows for one target."""
        start = 0
        for name, k in TARGET_CLASSES.items():
            if name == target:
                return slice(start, start + k)
            start += k
        raise ConfigError(f"unknown target {target!r}; known: {sorted(TARGET_CLASSES)}")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.out(self.trunk(x)).split(tuple(TARGET_CLASSES.values()), dim=-1)
        return dict(zip(TARGET_CLASSES, logits))


class Heads(nn.Module):
    """The now and next-hour heads.

    Args:
        d_model: Width of the decoder output they read (64).
        hidden: Width of each head's hidden layer (64), or 0 for a linear head.
        dropout: Dropout after the hidden layer (0.3).
        horizons: Which horizons to predict, from core/targets.py's HORIZONS (both).
    """

    def __init__(self, d_model: int = 64, hidden: int = 64, dropout: float = 0.3,
                 horizons: Sequence[str] = tuple(HORIZONS)):
        super().__init__()
        if isinstance(d_model, bool) or not isinstance(d_model, int) or d_model < 1:
            raise ConfigError(f"heads d_model must be a positive integer, not {d_model!r}")
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 0:
            raise ConfigError(f"heads hidden must be an integer of 0 or more, not {hidden!r}")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ConfigError(f"heads dropout must be at least 0 and below 1, not {dropout!r}")
        horizons = tuple(horizons)
        if not horizons or len(set(horizons)) != len(horizons) or set(horizons) - set(HORIZONS):
            raise ConfigError(f"heads horizons must be distinct names from {sorted(HORIZONS)}, not {horizons!r}")
        self.d_model = d_model
        self.heads = nn.ModuleDict({h: Head(d_model, hidden, dropout) for h in horizons})
        self.apply(init_weights)

    @property
    def labels(self) -> tuple:
        """Every output's label, horizon by horizon: ("sleep_now", "state_now", ...)."""
        return tuple(label_name(t, h) for h in self.heads for t in TARGET_CLASSES)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """(..., d_model) -> {label: logits (..., classes)}."""
        if x.dim() < 1 or x.shape[-1] != self.d_model:
            raise ModelError(f"heads read (..., {self.d_model}), not {tuple(x.shape)}")
        return {label_name(target, horizon): logits
                for horizon, head in self.heads.items() for target, logits in head(x).items()}

    @torch.no_grad()
    def set_priors(self, frequencies: Mapping[str, Sequence[float]], weight_scale: float = 0.01) -> None:
        """Start every output predicting the training hours' class mix.

        Args:
            frequencies: {target: class counts or frequencies}, such as {"sleep": [...2], "state": [...7]}.
                Every horizon of a target gets the same. Targets left out stay as they are.
            weight_scale: The output weights are shrunk to this fraction and the biases set to
                the log frequencies, so before training each output's probabilities are almost
                exactly the mix. They aren't zeroed: zero output weights would pass no gradient
                back to the layers before them on the first step.

        Raises:
            ConfigError: an unknown target, the wrong number of classes, a frequency that isn't
                positive and finite, or a weight_scale that isn't positive.
        """
        if isinstance(weight_scale, bool) or not isinstance(weight_scale, (int, float)) or not weight_scale > 0:
            raise ConfigError(f"weight_scale must be a positive number, not {weight_scale!r}")
        priors = {}
        for target, freq in frequencies.items():
            if target not in TARGET_CLASSES:
                raise ConfigError(f"unknown target {target!r}; known: {sorted(TARGET_CLASSES)}")
            freq = torch.as_tensor(freq, dtype=torch.float64)
            if freq.shape != (TARGET_CLASSES[target],) or not (torch.isfinite(freq).all() and (freq > 0).all()):
                raise ConfigError(f"{target} priors need {TARGET_CLASSES[target]} positive frequencies")
            priors[target] = (freq / freq.sum()).log()
        for head in self.heads.values():
            for target, log_freq in priors.items():
                rows = head.rows(target)
                head.out.weight[rows] *= weight_scale
                head.out.bias[rows] = log_freq.to(head.out.bias)


def probabilities(logits: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """{label: logits} -> {label: probabilities}, each row adding up to 1, in float32 or wider."""
    return {label: torch.softmax(z.float() if z.dtype in (torch.float16, torch.bfloat16) else z, dim=-1)
            for label, z in logits.items()}


__all__ = ["Head", "Heads", "TARGET_CLASSES", "probabilities"]
