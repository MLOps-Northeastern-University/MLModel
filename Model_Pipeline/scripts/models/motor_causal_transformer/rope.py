"""Positions by real time: rotary position embedding (RoPE) on each window's timestamp.

Attention needs to know how far apart two windows are. Counting tokens would be wrong here:
when the watch isn't worn, the next window can be hours later, yet it would sit one step
away. So each window's position is its real time, ``time_s / step_s``: back-to-back 10-s
windows are one step apart, and a gap in wear is a jump of as many steps as it lasted.

RoPE (Su et al. 2021, "RoFormer", arXiv:2104.09864) rotates each pair of numbers in a query
or key by an angle proportional to its position, with one frequency per pair:

    theta_i = base ** (-2i / head_dim)                      i = 0 .. head_dim/2 - 1
    angle_i = (time_s / step_s) * theta_i
    (a, b) -> (a cos(angle_i) - b sin(angle_i),  a sin(angle_i) + b cos(angle_i))

Pairs are adjacent numbers (0 and 1, 2 and 3, ...), as in the paper. Because a rotation by
angle p followed by the inverse of a rotation by angle q is a rotation by p - q, the dot
product of a rotated query and a rotated key depends only on how far apart their times are,
never on the times themselves. So a window's key can be rotated once and kept.

With the model's settings (head_dim 16: d_model 64 over 4 heads; base 10,000; step 10 s) the
8 pairs have wavelengths 2 pi step / theta_i of about 63 s, 3.3 min, 10.5 min, 33 min, 1.7 h,
5.5 h, 17 h and 55 h. The fastest tells neighbouring windows apart; the slowest turns once
in 55 h, so across the 8 h attention reach no two distances look alike.

How it runs fast (measured on an RTX 4060 Laptop GPU, batch 4 x 3,240 windows, q and k of
one layer, forward and backward): working out the angles costs about as much as rotating, so
``rotation`` does it once per batch and every layer reuses it. The rotation itself is one
complex multiplication, pair (a, b) as a + ib times e^(i angle), which runs as a single
kernel: 0.29 ms, against 0.49 ms as separate real multiply-adds and 0.55 ms recomputing the
angles on every call. Queries and keys can be rotated in the same call, stacked.

Precision: angles are computed in float64 and the rotation in float32 (complex64), then cast
back to the input's dtype. Times keep growing for as long as a patient wears the watch, and a
float32 angle of N radians is off by about N x 6e-8 radians: after 30 days the fastest pair
(N = 259,200) would be off by 0.016 radians, and it only gets worse. In float64 (about
N x 1e-16) it stays below 1e-8 radians even after 10 years. Rotating in bf16 (under mixed
precision) would lose about 0.4% of every number, so the rotation never runs in bf16.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn

from scripts.core.errors import ConfigError, ModelError


@dataclass(frozen=True)
class Rotation:
    """The rotations for one batch of tokens, worked out once and shared by every layer.

    Attributes:
        turn: e^(i angle) per token and pair, complex: (*time_shape, head_dim/2), where
            time_shape is the shape of the times it was made from, such as (batch, seq).
    """

    turn: torch.Tensor

    @property
    def time_shape(self) -> torch.Size:
        return self.turn.shape[:-1]


class TimeRoPE(nn.Module):
    """Rotates queries and keys by their tokens' real times.

    Use ``rotation`` once per batch, then ``rotate`` for each layer's queries and keys;
    calling the module does both at once, for a single use.

    Layout: the time dimensions lead. For times of shape (batch, seq), x is
    (batch, seq, ..., head_dim), such as (batch, seq, heads, head_dim), or
    (batch, seq, 2, heads, head_dim) for queries and keys stacked. That is the layout
    straight out of the q/k/v projection, before heads are moved forward for attention.

    It has no trained weights, and nothing in it is saved with the model: the frequencies
    are rebuilt from ``head_dim``, ``base`` and ``step_s``, which are model settings.

    Args:
        head_dim: Numbers per attention head. Must be even: RoPE rotates pairs.
        base: Sets how fast the slowest pair turns (larger is slower). Must be above 1.
        step_s: Seconds that count as one position step; the window length.
    """

    def __init__(self, head_dim: int, base: float = 10_000.0, step_s: float = 10.0):
        super().__init__()
        if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim < 2 or head_dim % 2:
            raise ConfigError(f"RoPE rotates pairs of numbers, so head_dim must be an even integer >= 2, "
                              f"not {head_dim!r}")
        if isinstance(base, bool) or not isinstance(base, (int, float)) or not (math.isfinite(base) and base > 1):
            raise ConfigError(f"RoPE base must be a finite number above 1, not {base!r}")
        if isinstance(step_s, bool) or not isinstance(step_s, (int, float)) or not (
                math.isfinite(step_s) and step_s > 0):
            raise ConfigError(f"RoPE step_s must be a finite number of seconds above 0, not {step_s!r}")
        self.head_dim, self.base, self.step_s = head_dim, float(base), float(step_s)
        # Angle per position step for each pair, in float64 (see Precision above). A buffer, so it
        # moves with .to(device); not persistent, so it isn't saved: it's rebuilt from the settings.
        theta = self.base ** (-torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
        self.register_buffer("theta", theta, persistent=False)

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, base={self.base:g}, step_s={self.step_s:g}"

    def wavelengths_s(self) -> torch.Tensor:
        """Each pair's wavelength in seconds: how long its angle takes to turn once, fastest first."""
        return 2 * math.pi * self.step_s / self.theta

    def rotation(self, time_s: torch.Tensor, dtype: torch.dtype = torch.float32) -> Rotation:
        """Work out the rotations for a batch of tokens, once.

        Args:
            time_s: The tokens' times in seconds, finite, such as (batch, seq). Any origin
                works, since only differences matter, but it must be the same origin for every
                token that attends to another.
            dtype: The precision ``rotate`` works in: float32 (default), or float64 for a
                float64 model.

        Returns:
            The batch's Rotation, on the same device as this module.
        """
        if dtype not in (torch.float32, torch.float64):
            raise ModelError(f"RoPE rotates in float32 or float64, not {dtype}")
        time_s = torch.as_tensor(time_s, device=self.theta.device)
        if time_s.dim() == 0:
            raise ModelError("time_s needs at least one dimension: one time per token")
        angles = (time_s.to(torch.float64) / self.step_s).unsqueeze(-1) * self.theta   # (*time, head_dim/2)
        turn = torch.polar(torch.ones_like(angles), angles)                             # e^(i angle), complex128
        return Rotation(turn.to(torch.complex64 if dtype == torch.float32 else torch.complex128))

    def rotate(self, x: torch.Tensor, rotation: Rotation) -> torch.Tensor:
        """Rotate queries or keys by a batch's rotations.

        Args:
            x: (*time_shape, ..., head_dim): queries, keys, or both stacked.
            rotation: From ``rotation``, for the same tokens.

        Returns:
            x rotated: the same shape and dtype.

        Raises:
            ModelError: x doesn't have head_dim numbers per head, or its leading dimensions
                aren't the rotation's tokens.
        """
        time_shape = rotation.time_shape
        if x.dim() <= len(time_shape) or x.shape[-1] != self.head_dim:
            raise ModelError(f"RoPE was built for {self.head_dim} numbers per head and tokens of shape "
                             f"{tuple(time_shape)}, got x of shape {tuple(x.shape)}")
        if x.shape[:len(time_shape)] != time_shape:
            raise ModelError(f"x of shape {tuple(x.shape)} doesn't start with the rotation's tokens, "
                             f"{tuple(time_shape)}")
        # One rotation per token, shared by every head (and by q and k when stacked): put a size-1
        # dimension in for each of x's dimensions between the tokens and the pairs.
        between = x.dim() - 1 - len(time_shape)
        turn = rotation.turn.view(*time_shape, *([1] * between), self.head_dim // 2)
        work = torch.float32 if turn.dtype == torch.complex64 else torch.float64
        pairs = x.to(work).unflatten(-1, (self.head_dim // 2, 2))   # (..., head_dim/2, 2): a and b of each pair
        try:
            z = torch.view_as_complex(pairs)                       # a + ib, no copy
        except RuntimeError:                                       # strides a complex view can't use
            z = torch.view_as_complex(pairs.contiguous())
        return torch.view_as_real(z * turn).flatten(-2).to(x.dtype)

    def forward(self, x: torch.Tensor, time_s: torch.Tensor) -> torch.Tensor:
        """Rotate x by its tokens' times in one go: ``rotate(x, rotation(time_s))``.

        For a model, prefer ``rotation`` once per batch and ``rotate`` per layer.
        """
        work = torch.float64 if x.dtype == torch.float64 else torch.float32
        return self.rotate(x, self.rotation(time_s, work))
