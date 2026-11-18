"""Pretraining: from the windows so far, predict the movement ahead. No diary answers are graded.

The decoder reads each chunk causally (a window sees only the windows before it, as live), and
every window predicts its future movement: the standardised harnet10 features ahead of it. This
teaches the movement encoder, token builder and decoder how movement unfolds over hours, around
doses and check-ins, from every worn hour, not just the diary's. The diary step then puts the
heads on top (network.py).

Five targets, each with its own prediction head (LayerNorm, Linear d_model -> features). The
loss is the mean of their mean squared errors:

    window, 20 s ahead      the first window 20-40 s later       fast, local detail (tremor rhythm)
    window, 2 min ahead     the first window 2-3 min later       activity patterns
    window, 10 min ahead    the first window 10-12 min later     slower context
    average, 30-90 min      the worn windows 30-90 min later     a dose working or wearing off
    average, 90-150 min     the worn windows 90-150 min later    the next dose due, the last fading

A single window an hour away is too random to predict, so the long targets are an hour's average.
It needs at least min_windows worn windows (180: half the hour). A window with no target gets no
loss: a gap, the chunk's end, padding, or an hour too little worn.

The averages need up to 2.5 h after the chunk. Those windows are only read to build targets and
are never fed to the model: ``future`` is the same sequences, run on past the chunk.

Each target has a bar to beat, logged alongside its loss (no gradient):
    windows    "it stays the same": the window's own features
    averages   "the next hours look like the last": the average of the hour before

Inputs. Check-ins are an ordinary input here as everywhere (tokens.py): a few true answers,
scattered through the past. The data step marks them, from the fold's training patients only.

Defaults are the final recipe's (pretraining run 181900): the horizons above, min_windows 180.

Changes from the prototype, and why:
- Padding can't lose a target. For the window targets, the prototype searched times padded with
  0, which aren't sorted. On random padded rows it found no target for 43% of the windows that
  had one (none were wrong, only missing), so pretraining had less signal than it should. Here
  padding sorts after every real window, so a padded row gets exactly the targets it gets alone.
  The averages were already right.
- No GPU sync: the parts come back as tensors, and parts_as_floats reads them in one go.
- One cumulative sum serves every average and the bar, instead of one each.

Speed. The loss never builds the heads' 1,024 guesses per window. The squared error is expanded
so that each window's work is 64 numbers (PredictionHead.squared_error). It is the same sum,
exactly, so the loss and gradients are the same. Every target is shrunk to 65 numbers before
it's looked up. Heads and loss, forward + backward, 4 x 3,240 windows with 2.5 h of future
(RTX 4060): 30 ms, against 40 ms building the guesses and 56 ms for the prototype.

Precision. Everything here runs in float32, or float64 when given it. The expanded square
subtracts sums of about the same size, which bf16 would round away. Against the plain squared
error it agrees to about 1e-10 in float64 (a test). The averages are differences of float32
cumulative sums: over a 10.5 h sequence, that is an error of about 1e-6 in a mean.
"""

import math
from typing import Dict, NamedTuple, Optional, Sequence, Tuple

import torch
from torch import nn

from model_pipeline.core.errors import ConfigError, ModelError
from model_pipeline.models.motor_causal_transformer.init import init_weights
from model_pipeline.models.motor_causal_transformer.movement import Harnet10Projection

WINDOWS = ((20, 20), (120, 60), (600, 120))     # (seconds ahead, slack): the first window in [ahead, ahead + slack]
AVERAGES = ((1800, 5400), (5400, 9000))         # (from, to) seconds ahead: the worn windows' mean in [from, to)
MIN_WINDOWS = 180
LAST_HOUR = (-3600, 0)                          # the averages' bar: the hour before


class Part(NamedTuple):
    """One target's share, as detached tensors for logging (except loss, which carries the gradient)."""

    loss: torch.Tensor     # mean squared error over the windows with a target
    bar: torch.Tensor      # the simple bar's error over the same windows (NaN if none had both)
    windows: torch.Tensor  # how many windows had a target


class ForecastLoss(NamedTuple):
    total: torch.Tensor              # the mean of the parts' losses, to backpropagate
    parts: Dict[str, Part]


class Future(NamedTuple):
    """The same sequences as the batch, run on past the chunk, for the averages' targets.

    Each row starts with exactly the batch row's real windows, then the windows after them.
    """

    features: torch.Tensor   # (batch, seq_future, features) standardised
    time_s: torch.Tensor     # (batch, seq_future)
    valid: torch.Tensor      # (batch, seq_future) bool, False for padding after the real windows


def harnet_targets(encoder: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """harnet10 features standardised by the encoder's own fitted standardisation: the targets.

    Raises:
        ConfigError: the encoder isn't harnet10, so it has no feature standardisation to predict in.
    """
    if not isinstance(encoder, Harnet10Projection):
        raise ConfigError(f"pretraining predicts standardised harnet10 features; the {encoder.name!r} "
                          "encoder has none, so its targets must be given another way")
    return torch.addcmul(encoder.shift, features.float(), encoder.scale)


def sorted_times(time_s: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """float64 times with padding at +inf, so they increase along each row and searches skip padding."""
    return torch.where(valid, time_s.to(torch.float64), math.inf).contiguous()


def window_ahead(time_s, valid, ahead_s, slack_s):
    """Each window's target: the first worn window from ahead_s to ahead_s + slack_s later.

    Returns:
        (index (batch, seq) into the row, has (batch, seq) bool).
    """
    t = sorted_times(time_s, valid)
    j = torch.searchsorted(t, t + ahead_s)                 # the first window at or after t + ahead_s
    j_safe = j.clamp_max(t.shape[1] - 1)
    found = torch.gather(t, 1, j_safe)
    return j_safe, valid & (j < t.shape[1]) & (found <= t + ahead_s + slack_s)


def window_averages(features, time_s, valid, spans, min_windows, queries=None):
    """Each window's mean of the worn windows from start_s to end_s after it (negative: before), per span.

    Args:
        features: (batch, seq, F) float32.
        spans: [(start_s, end_s)], each window's span [t + start_s, t + end_s).
        queries: Only the first this many windows need a mean (the batch's, not its future's).
            None: every window.

    Returns:
        [(mean (batch, queries, F), has (batch, queries): at least min_windows were there)], one per span.
    """
    q = time_s.shape[1] if queries is None else queries
    t = sorted_times(time_s, valid)
    total = torch.cumsum(features * valid.unsqueeze(-1), dim=1)
    total = torch.cat([torch.zeros_like(total[:, :1]), total], dim=1)            # (b, seq + 1, F)
    count = torch.cat([torch.zeros_like(valid[:, :1], dtype=torch.long), valid.long().cumsum(1)], dim=1)
    d = features.shape[-1]
    out = []
    for start_s, end_s in spans:
        lo = torch.searchsorted(t, t[:, :q] + start_s)                          # first at or after the start
        hi = torch.searchsorted(t, t[:, :q] + end_s)                            # first at or after the end
        n = torch.gather(count, 1, hi) - torch.gather(count, 1, lo)
        s = (torch.gather(total, 1, hi.unsqueeze(-1).expand(-1, -1, d))
             - torch.gather(total, 1, lo.unsqueeze(-1).expand(-1, -1, d)))
        out.append((s / n.clamp_min(1).unsqueeze(-1), valid[:, :q] & (n >= min_windows)))
    return out


def _masked_mean(x, mask):
    """Mean of x over mask; 0 if mask is empty (no sync to find out)."""
    return torch.where(mask, x, 0.0).sum() / mask.sum().clamp_min(1)


class PredictionHead(nn.Module):
    """A window's decoder output -> a guess at the future's standardised features: LayerNorm, Linear.

    The loss never calls forward (see squared_error); it's here to look at a head's guesses.
    """

    def __init__(self, d_model: int, feature_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, feature_dim)
        self.apply(init_weights)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(h))

    def project(self, targets: torch.Tensor) -> torch.Tensor:
        """(..., feature_dim) targets y -> (..., d_model + 1): [W^T y, c . y], what squared_error reads."""
        return targets @ torch.cat([self.linear.weight, self.linear.bias[:, None]], dim=1)

    def squared_error(self, z: torch.Tensor, projected: torch.Tensor, target_sq: torch.Tensor) -> torch.Tensor:
        """Mean over features of (prediction - target)^2, without building the prediction.

        The prediction is p = W z + c (W: feature_dim x d_model). Expanding the square,

            sum (p - y)^2 = z^T (W^T W) z + 2 (W^T c) . z + c . c     sum p^2
                          - 2 (z . W^T y + c . y)                       2 sum p y
                          + sum y^2

        is the same sum, so the loss and its gradients are the same. But every window's work is
        now 64 numbers, not 1,024. W^T W is d_model x d_model, worked out once.

        Args:
            z: (..., d_model) the LayerNorm'd decoder output, float32 or wider.
            projected: (..., d_model + 1) project(y).
            target_sq: (...) sum y^2.
        """
        w, c = self.linear.weight, self.linear.bias
        pp = ((z @ (w.T @ w)) * z).sum(-1) + 2 * (z @ (w.T @ c)) + c @ c
        py = (z * projected[..., :-1]).sum(-1) + projected[..., -1]
        # Rounding can take an almost-perfect guess a hair below 0; the true value can't be.
        return (pp - 2 * py + target_sq).clamp_min(0) / w.shape[0]


class MovementForecast(nn.Module):
    """The pretraining heads and their loss.

    Args:
        d_model: Width of the decoder output (64).
        feature_dim: Features per window to predict (1,024 for harnet10).
        windows: (seconds ahead, slack) per single-window target.
        averages: (from, to) seconds ahead per average target.
        min_windows: Worn windows an average needs (180).
    """

    def __init__(self, d_model: int = 64, feature_dim: int = 1024, windows: Sequence = WINDOWS,
                 averages: Sequence = AVERAGES, min_windows: int = MIN_WINDOWS):
        super().__init__()
        for name, value in (("d_model", d_model), ("feature_dim", feature_dim), ("min_windows", min_windows)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError(f"pretraining {name} must be a positive integer, not {value!r}")
        self.windows = tuple(_spans(windows, "windows", lambda a, b: a > 0 and b >= 0))
        self.averages = tuple(_spans(averages, "averages", lambda a, b: 0 <= a < b))
        if not self.windows and not self.averages:
            raise ConfigError("pretraining needs at least one target")
        self.names = tuple([f"ahead_{a:g}s" for a, _ in self.windows]
                           + [f"average_{a / 60:g}_{b / 60:g}min" for a, b in self.averages])
        self.d_model, self.feature_dim, self.min_windows = d_model, feature_dim, min_windows
        self.heads = nn.ModuleDict({name: PredictionHead(d_model, feature_dim) for name in self.names})

    @property
    def future_s(self) -> float:
        """How far past a chunk ``future`` must run: the furthest average's end (0 without averages)."""
        return max((b for _, b in self.averages), default=0.0)

    def forward(self, h: torch.Tensor, features: torch.Tensor, time_s: torch.Tensor, valid: torch.Tensor,
                graded: Optional[torch.Tensor] = None, future: Optional[Future] = None) -> ForecastLoss:
        """The pretraining loss of one batch.

        Args:
            h: (batch, seq, d_model) the decoder's output (MotorNetwork.encode).
            features: (batch, seq, feature_dim) the batch's standardised features (harnet_targets).
            time_s, valid: The batch's (Batch.time_s, Batch.valid).
            graded: (batch, seq) bool, which windows count, such as not the input-only past in front
                of a chunk. None: every real window.
            future: The sequences run on past the chunk, for the averages. Needed if there are any.

        Raises:
            ModelError: shapes that don't match, or averages without ``future``.
        """
        b, t = time_s.shape
        if h.shape != (b, t, self.d_model) or features.shape != (b, t, self.feature_dim) or valid.shape != (b, t):
            raise ModelError(f"pretraining needs h (b, t, {self.d_model}), features (b, t, {self.feature_dim}) "
                             f"and valid (b, t) for time_s {(b, t)}, not {tuple(h.shape)}, "
                             f"{tuple(features.shape)} and {tuple(valid.shape)}")
        graded = valid if graded is None else graded & valid
        if self.averages:
            if future is None:
                raise ModelError("the average targets need `future`: the sequences run on past the chunk")
            fb, ft = future.time_s.shape
            shapes_fit = future.features.shape == (fb, ft, self.feature_dim) and future.valid.shape == (fb, ft)
            if fb != b or ft < t or not shapes_fit:
                raise ModelError(f"future must be the batch's {b} rows, at least {t} windows long, with "
                                 f"{self.feature_dim} features, not {tuple(future.features.shape)}")
        # All in float32 (float64 if given): the expanded square subtracts sums of about the same
        # size, which bf16 would round away.
        with torch.autocast(device_type=h.device.type, enabled=False):
            h, features = _at_least_float32(h), _at_least_float32(features)
            parts = {}
            if self.windows:
                features_sq = features.square().sum(-1)                              # (b, t)
            for name, (ahead, slack) in zip(self.names, self.windows):
                head = self.heads[name]
                j, has = window_ahead(time_s, valid, ahead, slack)
                has = has & graded
                # Project every window once, then look the targets up: 65 numbers each, not 1,024.
                projected = torch.gather(head.project(features), 1, j.unsqueeze(-1).expand(-1, -1, self.d_model + 1))
                error = head.squared_error(head.norm(h), projected, torch.gather(features_sq, 1, j))
                with torch.no_grad():                                                # "it stays the same"
                    target = torch.gather(features, 1, j.unsqueeze(-1).expand(-1, -1, self.feature_dim))
                    bar_error = (features - target).square().mean(-1)
                parts[name] = _part(error, has, bar_error, has)
            if self.averages:
                spans = (LAST_HOUR, *self.averages)
                (last, has_last), *averages = window_averages(_at_least_float32(future.features), future.time_s,
                                                              future.valid, spans, self.min_windows, queries=t)
                for name, (target, has) in zip(self.names[len(self.windows):], averages):
                    head = self.heads[name]
                    has = has & graded
                    error = head.squared_error(head.norm(h), head.project(target), target.square().sum(-1))
                    with torch.no_grad():                                            # "like the last hour"
                        bar_error = (last - target).square().mean(-1)
                    parts[name] = _part(error, has, bar_error, has & has_last)
            total = torch.stack([p.loss for p in parts.values()]).mean()
        return ForecastLoss(total, parts)


def _at_least_float32(x):
    return x if x.dtype == torch.float64 else x.float()


def _part(error, has, bar_error, has_bar):
    """One target's loss over the windows that have it, and the bar's error over its windows."""
    loss = _masked_mean(error, has)
    bar = torch.where(has_bar.any(), _masked_mean(bar_error, has_bar), math.nan)
    return Part(loss, bar.detach(), has.sum())


def parts_as_floats(parts: Dict[str, Part]) -> Dict[str, Dict[str, float]]:
    """{name: {"loss", "bar", "windows"}} as Python numbers, read from the GPU in one sync."""
    if not parts:
        return {}
    values = torch.stack([torch.stack([p.loss.detach().float(), p.bar.float(), p.windows.float()])
                          for p in parts.values()]).tolist()
    return {name: {"loss": a, "bar": c, "windows": int(n)} for name, (a, c, n) in zip(parts, values)}


def _spans(spans, name, ok) -> Tuple[Tuple[float, float], ...]:
    out = []
    for span in spans:
        try:
            a, b = (float(v) for v in span)
        except (TypeError, ValueError):
            raise ConfigError(f"pretraining {name} must be pairs of seconds, not {span!r}") from None
        if not (math.isfinite(a) and math.isfinite(b) and ok(a, b)):
            raise ConfigError(f"pretraining {name}: {span!r} isn't a valid span")
        out.append((a, b))
    return tuple(out)


__all__ = ["AVERAGES", "MIN_WINDOWS", "WINDOWS", "ForecastLoss", "Future", "MovementForecast", "Part",
           "PredictionHead", "harnet_targets", "parts_as_floats", "window_ahead", "window_averages"]
