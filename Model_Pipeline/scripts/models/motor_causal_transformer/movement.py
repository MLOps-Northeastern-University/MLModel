"""Movement encoders: each window's movement in, an embedding of ``output_dim`` numbers out.

The rest of the model only sees the embedding, so the encoder can be swapped without touching
the tokens, decoder or heads. Each encoder says which per-window input it reads
(``input_group``: cached ``harnet`` features, or the raw ``signal``), fits whatever it needs
from the training windows (``fit``), and is registered under a name that config picks:

    movement:
      encoder: harnet10            # a name in ENCODERS
      params: {...}                # that encoder's own settings

To add one: subclass MovementEncoder, set ``name``, ``input_group`` and ``input_shape``, write
``forward`` (and ``fit`` if it learns anything from the data before training), decorate it with
``@register``, and add its tests. Its name is saved with the model, so a saved model always
rebuilds the encoder it was trained with.

harnet10 (the one in use)

    harnet10 (Yuan et al. 2024, "Self-supervised learning for human activity recognition using
    700,000 person-days of wearable data", npj Digital Medicine) is a ResNet pretrained on UK
    Biobank wrist accelerometry. It stays frozen: its 1,024 features per 10-s window are worked
    out once and cached (harnet.py, in this package), and this encoder reads the cache. Its
    licence is academic, non-commercial (Oxford's Academic Use Licence); commercial use needs
    another.

    standardise   (x - training mean) / training SD, per feature, so its scale doesn't depend
                  on the data; the mean and SD are fitted on the training windows only
    Dropout 0.5   1,024 numbers learnt elsewhere: keeps the model from leaning on a few
    Linear 1024 -> 128, ReLU
    Dropout 0.2
    Linear 128 -> 256, LayerNorm

    The same layers as the prototype's harnet branch and join.
"""

from abc import ABC, abstractmethod
from typing import ClassVar, Dict, Iterable, Tuple, Type, Union

import numpy as np
import torch
from torch import nn

from scripts.core.errors import ConfigError, ModelError, NotFittedError
from scripts.models.motor_causal_transformer.init import init_weights

ENCODERS: Dict[str, Type["MovementEncoder"]] = {}
INPUT_GROUPS = ("harnet", "signal")


def register(cls: Type["MovementEncoder"]) -> Type["MovementEncoder"]:
    """Add an encoder class to ENCODERS under its name (use as a decorator)."""
    if not getattr(cls, "name", None) or cls.input_group not in INPUT_GROUPS:
        raise ConfigError(f"{cls.__name__} needs a name and an input_group from {INPUT_GROUPS}")
    if cls.name in ENCODERS and ENCODERS[cls.name] is not cls:
        raise ConfigError(f"a movement encoder named {cls.name!r} is already registered")
    ENCODERS[cls.name] = cls
    return cls


def build_movement_encoder(name: str, params: Dict = None) -> "MovementEncoder":
    """The encoder config names, with its params.

    Raises:
        ConfigError: no encoder by that name, or params it doesn't take.
    """
    if name not in ENCODERS:
        raise ConfigError(f"movement encoder must be one of {sorted(ENCODERS)}, not {name!r}")
    try:
        return ENCODERS[name](**(params or {}))
    except TypeError as e:
        raise ConfigError(f"movement encoder {name!r}: {e}") from None


class MovementEncoder(nn.Module, ABC):
    """Base class: one window's movement input into ``output_dim`` numbers.

    Attributes:
        name: The name config uses, and the model saves.
        input_group: Which per-window input it reads: "harnet" or "signal".
        input_shape: One window's input, such as (1024,) for harnet10's features.
        output_dim: Numbers per window it gives (set by the constructor).
    """

    name: ClassVar[str] = ""
    input_group: ClassVar[str] = ""
    input_shape: ClassVar[Tuple[int, ...]] = ()
    output_dim: int

    def fit(self, inputs: Iterable[np.ndarray]) -> None:
        """Learn what it needs from the training windows' inputs, before training. Default: nothing."""

    @property
    def is_fitted(self) -> bool:
        return True

    def check_input(self, x: torch.Tensor) -> None:
        """Raise ModelError unless x is (N, *input_shape)."""
        if x.dim() != 1 + len(self.input_shape) or tuple(x.shape[1:]) != tuple(self.input_shape):
            raise ModelError(f"{self.name} reads ({', '.join(['N', *map(str, self.input_shape)])}) "
                             f"{self.input_group}, not {tuple(x.shape)}")

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, *input_shape) -> (N, output_dim)."""


def _check_dim(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"movement encoder {name} must be a positive integer, not {value!r}")


def _check_dropout(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
        raise ConfigError(f"movement encoder {name} must be at least 0 and below 1, not {value!r}")


@register
class Harnet10Projection(MovementEncoder):
    """harnet10's cached 1,024 features per window, standardised and projected to output_dim.

    Args:
        hidden_dim: Width of the first projection (128).
        output_dim: Numbers per window it gives (256).
        input_dropout: Dropout on the standardised features (0.5).
        dropout: Dropout between the two projections (0.2).
    """

    name = "harnet10"
    input_group = "harnet"
    input_shape = (1024,)
    MIN_SD = 1e-6   # a feature that never varies in training is centred, not blown up
    FIT_BLOCK = 16_384   # windows summed at a time when fitting: about 130 MB in float64

    def __init__(self, hidden_dim: int = 128, output_dim: int = 256, input_dropout: float = 0.5,
                 dropout: float = 0.2):
        super().__init__()
        _check_dim("hidden_dim", hidden_dim)
        _check_dim("output_dim", output_dim)
        _check_dropout("input_dropout", input_dropout)
        _check_dropout("dropout", dropout)
        self.output_dim = output_dim
        features = self.input_shape[0]
        # Standardisation, saved with the model: x * scale + shift = (x - mean) / sd, one fused
        # multiply-add instead of a subtract and a divide.
        self.register_buffer("scale", torch.ones(features))
        self.register_buffer("shift", torch.zeros(features))
        self.register_buffer("fitted", torch.tensor(False))
        self._fitted_known = False   # a Python copy of `fitted`, so forward never waits on the GPU to read it
        self.project = nn.Sequential(
            nn.Dropout(input_dropout), nn.Linear(features, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim))
        self.apply(init_weights)

    @property
    def is_fitted(self) -> bool:
        # Read the buffer only until it says fitted: on a GPU each read would wait for the GPU.
        if not self._fitted_known:
            self._fitted_known = bool(self.fitted)
        return self._fitted_known

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._fitted_known = False   # the loaded buffer decides again

    def fit(self, inputs: Union[np.ndarray, torch.Tensor, Iterable[np.ndarray]]) -> None:
        """Fit the standardisation on the training windows' features.

        Args:
            inputs: (N, 1024) features, or an iterable of such chunks (such as one per
                participant), so all the training windows never need to be in memory at once.
                Summed in float64, so millions of windows lose no precision.

        Raises:
            ModelError: a chunk of the wrong shape, a feature that isn't finite, or fewer than
                2 windows in all.
        """
        chunks = [inputs] if isinstance(inputs, (np.ndarray, torch.Tensor)) else inputs
        n, total, squares = 0, np.zeros(self.input_shape[0]), np.zeros(self.input_shape[0])
        for chunk in chunks:
            chunk = chunk.cpu().numpy() if isinstance(chunk, torch.Tensor) else chunk
            if chunk.ndim != 2 or chunk.shape[1] != self.input_shape[0]:
                raise ModelError(f"harnet10 features must be (N, {self.input_shape[0]}), not {chunk.shape}")
            # A block at a time, so a patient's features are never copied whole into float64.
            for start in range(0, len(chunk), self.FIT_BLOCK):
                block = np.asarray(chunk[start:start + self.FIT_BLOCK], dtype=np.float64)
                if not np.isfinite(block).all():
                    raise ModelError("harnet10 features must all be finite")
                total += block.sum(axis=0)
                squares += np.square(block).sum(axis=0)
            n += len(chunk)
        if n < 2:
            raise ModelError(f"fitting harnet10's standardisation needs at least 2 windows, not {n}")
        mean = total / n
        var = np.maximum(squares - n * np.square(mean), 0.0) / (n - 1)   # the sample variance
        self.set_standardisation(mean, np.sqrt(var))

    def set_standardisation(self, mean, sd) -> None:
        """Set the standardisation from a mean and SD per feature worked out elsewhere."""
        mean, sd = (np.asarray(v, dtype=np.float64).reshape(-1) for v in (mean, sd))
        if mean.shape != self.input_shape or sd.shape != self.input_shape:
            raise ModelError(f"harnet10 standardisation needs {self.input_shape[0]} means and SDs")
        if not (np.isfinite(mean).all() and np.isfinite(sd).all() and (sd >= 0).all()):
            raise ModelError("harnet10 standardisation needs finite means and SDs of 0 or more")
        scale = 1.0 / np.maximum(sd, self.MIN_SD)
        self.scale.copy_(torch.as_tensor(scale, dtype=torch.float32))
        self.shift.copy_(torch.as_tensor(-mean * scale, dtype=torch.float32))
        self.fitted.fill_(True)
        self._fitted_known = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise NotFittedError("fit harnet10's standardisation on the training windows before using it")
        self.check_input(x)
        return self.project(torch.addcmul(self.shift, x, self.scale))


def describe(encoder: MovementEncoder) -> str:
    """One line for logs: the encoder, what it reads, and its size."""
    n = sum(p.numel() for p in encoder.parameters())
    shape = "x".join(map(str, encoder.input_shape))
    return f"{encoder.name}: {encoder.input_group} ({shape}) -> {encoder.output_dim}, {n:,} weights"


__all__ = ["ENCODERS", "INPUT_GROUPS", "MovementEncoder", "Harnet10Projection", "build_movement_encoder",
           "describe", "register"]
