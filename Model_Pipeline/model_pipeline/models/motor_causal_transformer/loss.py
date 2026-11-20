"""The training loss: each diary hour's answer against the model's predictions for that hour.

For every output (sleep, state and tremor, now and next hour) and every diary hour that has an
answer and worn windows:

    cost(prediction, answer) = cross-entropy + lam x expected distance
    hour term     = cost(the hour's pooled prediction)
    per-window    = the average over the hour's worn windows of cost(each window's own prediction)
    loss          = sum over outputs of label_weight x (hour_weight x hour term + beta x per-window)

Cross-entropy rewards probability on the diary's answer. The expected distance, the sum of
P(class) x |class score - answer score|, makes a near miss cost less than a far one, since the
scores are ordered (kinesia -3 .. +3, tremor 0 .. 2). lam says how much it counts. Sleep has no
scores, so its cost is cross-entropy only.

The per-window term (beta) grades every window against its hour's answer. It averages the
windows' costs, not their probabilities, so a confidently wrong window gets the strongest
correction, and windows the pooling ignores still learn. Without it, a low temperature could
leave most windows untrained.

Hours with no answer (NaN) or no worn windows are left out, never counted as 0. The state and
tremor have no answer in sleep hours (core/targets.py). An output with no answered hour in the
batch adds exactly 0.

Class weights make a rare class count as much as a common one. Each hour's terms are multiplied
by the weight of its true class, and the hours are averaged with those weights, so the loss
keeps its size. balanced_weights works them out from the training hours, like scikit-learn's
class_weight="balanced", with a cap so a class with a handful of hours can't take over. Before
set_class_weights every weight is 1, which is the plain mean.

Defaults are the final recipe's: lam 0.3, beta 0.2, hour_weight 1, balanced class weights with
power 1 and cap 10. The maths is the prototype's.

Speed:
- The cost of every possible answer is worked out at once: -log p + lam x (p @ D), where
  D[j, c] = |score j - score c|.
- The per-window term is then one matmul, (each hour's window weights) @ (every window's cost
  per class), and a gather of the hour's answer. The prototype instead copied every window's
  prediction once per hour (batch x hours x windows x classes).
- The distance matrices live on the device as buffers, not rebuilt on every call.
- No GPU sync: the parts for logging come back as tensors, and as_floats reads them all in one go.

Precision: the loss always runs in float32 or wider, even under autocast.
"""

import math
from typing import Dict, Mapping, NamedTuple, Optional, Sequence

import torch
from torch import nn

from model_pipeline.core.errors import ConfigError, ModelError
from model_pipeline.core.targets import TARGETS, parse_label

EPS = 1e-8   # the smallest probability a log is taken of, so a 0 can't give an infinite loss


class LabelLoss(NamedTuple):
    """One output's share of the loss, as detached tensors for logging."""

    hour: torch.Tensor      # the class-weighted mean hour term
    safety: torch.Tensor    # the class-weighted mean per-window term
    hours: torch.Tensor     # how many hours were graded


class LossResult(NamedTuple):
    total: torch.Tensor                 # the loss to backpropagate
    parts: Dict[str, LabelLoss]         # per output, for logging


def balanced_weights(counts: Sequence[float], power: float = 1.0, cap: float = 10.0) -> torch.Tensor:
    """Class weights from the training hours' class counts.

    (1 / frequency) ** power, at most cap times the smallest weight, scaled so they average 1
    over the training hours (so the loss keeps its size). power 1 is scikit-learn's "balanced",
    and power 0 gives no weighting. A class with no hours counts as 1, so its weight is capped,
    not infinite.

    Raises:
        ConfigError: counts that aren't one finite, non-negative number per class, a negative
            power, or a cap below 1.
    """
    counts = torch.as_tensor(counts, dtype=torch.float64)
    if counts.dim() != 1 or len(counts) < 1 or not (torch.isfinite(counts).all() and (counts >= 0).all()):
        raise ConfigError("class counts must be one finite number of 0 or more per class")
    if isinstance(power, bool) or not isinstance(power, (int, float)) or not 0 <= power < math.inf:
        raise ConfigError(f"class weight power must be a number of 0 or more, not {power!r}")
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not 1 <= cap < math.inf:
        raise ConfigError(f"class weight cap must be a number of 1 or more, not {cap!r}")
    freq = counts.clamp_min(1)
    freq = freq / freq.sum()
    w = freq ** -power
    w = torch.minimum(w, cap * w.min())
    return (w / (freq * w).sum()).float()


def _distance_matrix(scores):
    """D[j, c] = |score j - score c|, 0 wherever a class has no score."""
    has = torch.tensor([s is not None for s in scores])
    s = torch.tensor([0.0 if v is None else float(v) for v in scores])
    d = (s[:, None] - s[None, :]).abs()
    return d * (has[:, None] & has[None, :]), has


def _at_least_float32(x):
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


class MILLoss(nn.Module):
    """The MIL loss over every output with labels in the batch.

    Args:
        lam: How much the expected distance counts against cross-entropy (0.3).
        beta: How much the per-window term counts (0.2).
        hour_weight: How much the pooled hour term counts (1.0; 0 trains on the per-window term only).
        label_weights: Optional {label: weight}; 1 for every output by default.
    """

    def __init__(self, lam: float = 0.3, beta: float = 0.2, hour_weight: float = 1.0,
                 label_weights: Optional[Mapping[str, float]] = None):
        super().__init__()
        for name, value in (("lam", lam), ("beta", beta), ("hour_weight", hour_weight)):
            _check_weight(name, value)
        label_weights = dict(label_weights or {})
        for label, value in label_weights.items():
            parse_label(label)                                  # ConfigError for an unknown label
            _check_weight(f"label weight for {label}", value)
        self.lam, self.beta, self.hour_weight = float(lam), float(beta), float(hour_weight)
        self.label_weights = {k: float(v) for k, v in label_weights.items()}
        self._scored = {}       # target -> "all", "some" or "none" of its classes have a score
        for name, target in TARGETS.items():
            d, has = _distance_matrix(target.scores)
            self.register_buffer(f"distance_{name}", d, persistent=False)
            self.register_buffer(f"scored_{name}", has.float(), persistent=False)
            self._scored[name] = "all" if has.all() else "none" if not has.any() else "some"
            # Saved with the model, so a run can be resumed or audited with its weights.
            self.register_buffer(f"class_weights_{name}", torch.ones(len(target.classes)))

    def set_class_weights(self, weights: Mapping[str, Sequence[float]]) -> None:
        """{target: (classes,) weights}, such as from balanced_weights. Targets left out stay as they are."""
        for target, w in weights.items():
            if target not in TARGETS:
                raise ConfigError(f"unknown target {target!r}; known: {sorted(TARGETS)}")
            w = torch.as_tensor(w, dtype=torch.float32)
            buf = getattr(self, f"class_weights_{target}")
            if w.shape != buf.shape or not (torch.isfinite(w).all() and (w > 0).all()):
                raise ConfigError(f"{target} class weights must be {buf.numel()} positive numbers")
            buf.copy_(w)

    def class_weights(self, target: str) -> torch.Tensor:
        return getattr(self, f"class_weights_{target}")

    def expected_distance(self, probs: torch.Tensor, target: str) -> torch.Tensor:
        """(..., classes) probabilities -> (..., classes): the expected distance if each class were the answer.

        Taken over the classes that have a score, with the probabilities renormalised over them.
        It is 0 for an answer without a score.
        """
        d = getattr(self, f"distance_{target}").to(probs.dtype)
        if self._scored[target] == "none":
            return torch.zeros_like(probs)
        if self._scored[target] == "some":
            scored = getattr(self, f"scored_{target}").to(probs.dtype)
            probs = probs * scored
            probs = probs / probs.sum(-1, keepdim=True).clamp_min(EPS)
        return probs @ d

    def forward(self, pooled: Mapping, window_logits: Mapping[str, torch.Tensor], members: torch.Tensor,
                targets: Mapping[str, torch.Tensor], has_windows: torch.Tensor,
                sample_weights: Optional[torch.Tensor] = None) -> LossResult:
        """The loss over every output in targets.

        Args:
            pooled: {label: Pooled} from GatedPooling (its attention is the hour's prediction).
            window_logits: {label: (batch, seq, classes)} the heads' outputs per window.
            members: (batch, hours, seq) bool, which worn windows belong to each hour (hour_members).
            targets: {label: (batch, hours)} class indices as floats, NaN for no answer. A
                next_hour label holds the next hour's answers (core/targets.py, at_horizon).
                The class indices aren't checked here, which would need a GPU sync: they come
                from core/targets.py, which checks them.
            has_windows: (batch, hours) bool, False for hours with no worn windows.
            sample_weights: (batch, hours) each hour's weight (bias mitigation by reweighting), or
                None for 1 each. It scales the hour's share of the loss, which is normalised by
                the class weights alone: so a batch of low-weight hours weighs less than one of
                high-weight hours, as reweighting needs (fit rescales the weights to average 1,
                so the loss keeps its size), and an hour of weight 0 learns nothing.

        Returns:
            LossResult(total, {label: LabelLoss}).

        Raises:
            ConfigError: an unknown label.
            ModelError: a label with no predictions, or shapes that don't match.
        """
        if members.dim() != 3 or has_windows.shape != members.shape[:2]:
            raise ModelError(f"members must be (batch, hours, seq) and has_windows (batch, hours), not "
                             f"{tuple(members.shape)} and {tuple(has_windows.shape)}")
        if sample_weights is not None and sample_weights.shape != has_windows.shape:
            raise ModelError(f"sample_weights must be {tuple(has_windows.shape)}, not {tuple(sample_weights.shape)}")
        with torch.autocast(device_type=members.device.type, enabled=False):
            terms, window_weights, parts = [], None, {}
            for label, answers in targets.items():
                target, _ = parse_label(label)
                if label not in pooled or label not in window_logits:
                    raise ModelError(f"no predictions for {label}")
                if answers.shape != has_windows.shape:
                    raise ModelError(f"{label} targets must be {tuple(has_windows.shape)}, not {tuple(answers.shape)}")
                hour_probs = _at_least_float32(pooled[label].attention)
                logits = _at_least_float32(window_logits[label])
                if window_weights is None:   # each hour's windows, equally weighted: shared by every output
                    window_weights = members.to(logits.dtype) / members.sum(-1, keepdim=True).clamp_min(1)
                valid = has_windows & ~torch.isnan(answers)
                y = torch.where(valid, answers, 0).long().unsqueeze(-1)                  # (b, H, 1)

                # Every class's cost at once, then the answer's.
                hour_cost = -hour_probs.clamp_min(EPS).log()
                log_p = torch.log_softmax(logits, dim=-1)                                 # (b, t, k)
                window_cost = -log_p
                if self.lam:
                    hour_cost = hour_cost + self.lam * self.expected_distance(hour_probs, target)
                    window_cost = window_cost + self.lam * self.expected_distance(log_p.exp(), target)
                hour_term = hour_cost.gather(-1, y).squeeze(-1)                           # (b, H)
                safety_term = (window_weights @ window_cost).gather(-1, y).squeeze(-1)   # (b, H)

                cw = torch.where(valid, self.class_weights(target).to(logits.dtype)[y.squeeze(-1)], 0.0)
                # The norm is the class weights' only: dividing by the sample-weighted sum would make
                # every batch weigh the same, whatever its hours' weights.
                norm = cw.sum().clamp_min(EPS)                # 0 hours: every term below is exactly 0
                w = cw if sample_weights is None else cw * sample_weights.to(cw.dtype)
                hour_mean = torch.where(valid, hour_term, 0.0).mul(w).sum() / norm
                safety_mean = torch.where(valid, safety_term, 0.0).mul(w).sum() / norm
                weight = self.label_weights.get(label, 1.0)
                terms.append(weight * (self.hour_weight * hour_mean + self.beta * safety_mean))
                parts[label] = LabelLoss(hour_mean.detach(), safety_mean.detach(), valid.sum())
            total = torch.stack(terms).sum() if terms else members.new_zeros((), dtype=torch.float32)
        return LossResult(total, parts)


def as_floats(parts: Mapping[str, LabelLoss]) -> Dict[str, Dict[str, float]]:
    """{label: {"hour", "safety", "hours"}} as Python numbers, read from the GPU in one sync."""
    if not parts:
        return {}
    values = torch.stack([torch.stack([p.hour.float(), p.safety.float(), p.hours.float()])
                          for p in parts.values()]).tolist()
    return {label: {"hour": h, "safety": s, "hours": int(n)} for label, (h, s, n) in zip(parts, values)}


def _check_weight(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < math.inf:
        raise ConfigError(f"loss {name} must be a number of 0 or more, not {value!r}")


__all__ = ["EPS", "LabelLoss", "LossResult", "MILLoss", "as_floats", "balanced_weights"]
