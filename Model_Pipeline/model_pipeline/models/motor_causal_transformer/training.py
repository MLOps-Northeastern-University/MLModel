"""The two training steps, and scoring diary hours.

    pretrain(net, forecast, recordings, PretrainSettings(), device, rng, val)  step 1, no diary grading
    train_diary(net, train, val, DiarySettings(), device, rng)               step 2, heads and pooling
    score_hours(net, recordings, past_hours, device)                         each answered hour alone

Step 1, pretraining. The movement encoder, token builder and decoder learn to predict the movement
ahead (pretraining.py) from every worn hour of the training patients. Check-ins are in the input,
as everywhere. The heads and pooling don't take part. Each epoch logs every target's loss, its bar
(the simplest guess's error) and its skill, 1 - loss / bar: above 0 beats the simple guess. With
validation patients it also measures them, the movement only (no labels), with no gradients and the
same chunks every epoch, so the val/ skill compares one pretraining with another. That pass has its
own random numbers, so measuring never changes the training.

Step 2, the diary step. The pretrained parts are frozen (no dropout either, network.py), and the
heads and pooling learn the diary's answers. The heads start at the training hours' class mix,
and the loss weighs classes by their rarity (balanced, capped). With validation patients it keeps
the epoch with the best within-patient score (select_by: the mean of OFF, dyskinetic and any-tremor
AUC, or one of them) and stops after `patience` epochs without a better
one. Without them (the final fit), it trains a fixed number of epochs, chosen by cross-validation.

Validation scores every answered hour on its own, with exactly L hours of past in front of it,
for each L in past_hours (1, 2, 4 and 8 h), and averages over the Ls. That is the "mean
within-patient AUC": the mean of OFF, dyskinetic and any-tremor AUC, each worked out within every
patient and averaged over patients. This is immune to patients' different base rates. Validation
has no simulated check-ins unless asked.

Defaults are the final recipe's (pretraining run 181900, diary run 183136), except for three
choices made since, which need validating:
- check-ins everywhere, pretraining and past included (they were only on graded hours);
- no dropout in frozen parts (it was on);
- pretraining's padded rows keep their short-range targets (a bug fix, pretraining.py).
Weight decay now skips biases, norms and temperatures in pretraining too; the prototype's
pretraining decayed every weight.

Speed: a background thread builds the next batch (reading the features, check-ins, padding)
while the GPU runs this one. The parts for logging are read from the GPU once per epoch.
"""

import math
import queue
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from model_pipeline.core.errors import ConfigError, ModelError
from model_pipeline.core.targets import STATE_GROUPS
from model_pipeline.models.motor_causal_transformer.data import (
    HOUR, Recording, answered_hours, batches, collate, make_example, tile_chunks)
from model_pipeline.models.motor_causal_transformer.heads import TARGET_CLASSES
from model_pipeline.models.motor_causal_transformer.loss import balanced_weights
from model_pipeline.models.motor_causal_transformer.movement import Harnet10Projection
from model_pipeline.models.motor_causal_transformer.network import PARTS, MotorNetwork
from model_pipeline.models.motor_causal_transformer.pretraining import (
    Future, MovementForecast, Part, harnet_targets, parts_as_floats)
from model_pipeline.utils.logging_setup import get_logger

logger = get_logger(__name__)

PRETRAINED = ("movement", "tokens", "decoder")


# ---- settings ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class PretrainSettings:
    """Step 1's settings (pretraining run 181900, with check-ins added)."""

    epochs: int = 20
    chunk_h: int = 8
    past_h: int = 1                  # ungraded past in front of each chunk
    batch: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    check_ins_per_day: float = 4.0
    amp: bool = True                 # bf16 autocast on a GPU

    def __post_init__(self):
        _check(self)


@dataclass(frozen=True)
class DiarySettings:
    """Step 2's settings (diary run 183136, with check-ins in the past and no frozen dropout)."""

    max_epochs: int = 15
    patience: int = 3
    chunk_h: int = 8
    past_h: int = 1
    batch: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    feature_noise: float = 0.1       # Gaussian noise on harnet10's features, in SDs of each feature
    check_ins_per_day: float = 4.0
    class_weight_power: float = 1.0
    class_weight_cap: float = 10.0
    learn: Tuple[str, ...] = ("heads", "pooling")
    keep_dropout: bool = False       # frozen parts' dropout on, as the prototype had it
    eval_past_h: Tuple[int, ...] = (1, 2, 4, 8)
    eval_batch: int = 16
    amp: bool = True
    # The validation score that picks the epoch kept and drives early stopping: the mean within-patient
    # AUC, or one question's. OFF alone keeps the epoch best at OFF, which the mean can trade away for
    # dyskinetic or tremor.
    select_by: str = "within_auc"

    def __post_init__(self):
        if self.select_by not in SELECT_BY:
            raise ConfigError(f"select_by must be one of {list(SELECT_BY)}, not {self.select_by!r}")
        object.__setattr__(self, "learn", tuple(self.learn))
        object.__setattr__(self, "eval_past_h", tuple(self.eval_past_h))
        _check(self)
        unknown = set(self.learn) - set(PARTS)
        if not self.learn or unknown:
            raise ConfigError(f"learn must be parts from {list(PARTS)}, not {list(self.learn)}")
        if not self.eval_past_h or any(isinstance(h, bool) or not isinstance(h, int) or h < 0
                                       for h in self.eval_past_h):
            raise ConfigError(f"eval_past_h must be whole hours of 0 or more, not {list(self.eval_past_h)}")


SELECT_BY = ("within_auc", "off_auc_within", "dyskinetic_auc_within", "tremor_any_auc_within")


def _check(settings):
    for name, value in asdict(settings).items():
        if isinstance(value, (bool, str)) or isinstance(value, (tuple, list)):
            continue
        whole = name in ("epochs", "max_epochs", "patience", "chunk_h", "past_h", "batch", "eval_batch")
        lowest = 0 if name in ("past_h", "epochs") else 1          # 0 epochs: skip pretraining
        if whole and (not isinstance(value, int) or value < lowest):
            raise ConfigError(f"{name} must be a whole number of {lowest} or more, not {value!r}")
        if not whole and (not isinstance(value, (int, float)) or not 0 <= value < math.inf):
            raise ConfigError(f"{name} must be a number of 0 or more, not {value!r}")
    if getattr(settings, "lr", 1) <= 0:
        raise ConfigError("lr must be above 0")


# ---- step 1 ------------------------------------------------------------------------------------

def pretrain(net: MotorNetwork, forecast: MovementForecast, recordings: Dict[str, Recording],
             settings: PretrainSettings, device, rng: np.random.Generator,
             val: Optional[Dict[str, Recording]] = None) -> List[dict]:
    """Pretrain the movement encoder, token builder and decoder. Returns each epoch's log line:

        {"epoch", "seconds", "<target>/loss", "<target>/bar", "<target>/skill", "<target>/windows",
         and with val the same as "val/<target>/..."}

    The movement encoder must already be fitted (its standardisation sets the targets' scale).
    """
    if not net.movement.is_fitted:
        raise ModelError("fit the movement encoder's standardisation before pretraining")
    net.freeze_all_but(PRETRAINED)
    net.to(device).train()
    forecast.to(device).train()
    groups = net.parameter_groups(settings.weight_decay)
    groups[0]["params"] += [p for p in forecast.parameters() if p.ndim >= 2]
    groups[1]["params"] += [p for p in forecast.parameters() if p.ndim < 2]
    learning = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, lr=settings.lr)
    future_h = math.ceil(forecast.future_s / HOUR)
    pin, amp = _device_type(device) == "cuda", settings.amp and _device_type(device) == "cuda"
    history = []
    for epoch in range(1, settings.epochs + 1):
        started = time.perf_counter()
        parts = []
        for c in prefetch(_pretrain_batches(net, recordings, settings, future_h, rng, pin)):
            out = _forecast_step(net, forecast, c, device, pin, amp)
            opt.zero_grad(set_to_none=True)
            out.total.backward()
            torch.nn.utils.clip_grad_norm_(learning, settings.grad_clip)
            opt.step()
            parts.append({k: p._replace(loss=p.loss.detach()) for k, p in out.parts.items()})   # no graph kept
        line = {"epoch": epoch, **_with_skill(_mean_parts(parts))}
        if val:
            line.update({f"val/{k}": v for k, v in _with_skill(_forecast_error(
                net, forecast, val, settings, device, future_h, pin, amp)).items()})
        line["seconds"] = time.perf_counter() - started
        history.append(line)
        logger.info("pretrain epoch %d: %s", epoch, _describe(line), extra={"event": "pretrain_epoch", **line})
    return history


def _pretrain_batches(net, recordings, settings, future_h, rng, pin):
    """One epoch of pretraining batches: tiled chunks with their future, simulated check-ins in."""
    for group in batches(tile_chunks(recordings, settings.chunk_h, rng, answered_only=False), settings.batch, rng):
        examples = [make_example(recordings[pid], start, settings.chunk_h, settings.past_h, future_h,
                                 settings.check_ins_per_day, rng) for pid, start in group]
        examples = [e for e in examples if e.n_input > 1 and e.has_graded_windows]
        if examples:
            yield collate(examples, net.labels, future=True, pin=pin)


def _forecast_step(net, forecast, c, device, pin, amp):
    """The forecast's output for one collated batch: its total loss and every target's parts."""
    batch = c.batch.to(device, non_blocking=pin)
    f_features, f_time, f_valid = (x.to(device, non_blocking=pin) for x in c.future)
    future = Future(harnet_targets(net.movement, f_features), f_time, f_valid)
    with torch.autocast(_device_type(device), dtype=torch.bfloat16, enabled=amp):
        h = net.encode(batch)
        return forecast(h, harnet_targets(net.movement, batch.movement), batch.time_s, batch.valid,
                        graded=c.graded.to(device, non_blocking=pin), future=future)


# The validation pass's own seed: the same chunks and check-ins every epoch and every run, and the
# training's random numbers untouched.
VAL_SEED = 0


@torch.no_grad()
def _forecast_error(net, forecast, recordings, settings, device, future_h, pin, amp):
    """Each target's loss and bar on recordings, without learning: dropout off, the same chunks each time.

    No simulated check-ins: they would put these patients' own diary answers in the input, and the
    pass measures how well the movement alone is forecast (as the diary step's validation, too,
    has none).
    """
    rng = np.random.default_rng(VAL_SEED)
    movement_only = replace(settings, check_ins_per_day=0.0)
    net.eval()
    forecast.eval()
    try:
        parts = [_forecast_step(net, forecast, c, device, pin, amp).parts
                 for c in _pretrain_batches(net, recordings, movement_only, future_h, rng, pin)]
    finally:
        net.train()
        forecast.train()
    return _mean_parts(parts)


def _with_skill(line):
    """A {"<target>/loss", "<target>/bar", ...} line with each target's skill, 1 - loss / bar, added."""
    out = dict(line)
    for key, loss in line.items():
        if key.endswith("/loss"):
            name = key[:-len("/loss")]
            bar = line.get(f"{name}/bar", math.nan)
            out[f"{name}/skill"] = 1 - loss / bar if math.isfinite(bar) and bar > 0 else math.nan
    return out


def _mean_parts(parts):
    """Each target's mean loss and bar over an epoch's steps, read from the GPU once."""
    if not parts:
        return {}
    epoch = {name: Part(torch.stack([p[name].loss.detach() for p in parts]).mean(),
                        torch.stack([p[name].bar for p in parts]).nanmean(),
                        torch.stack([p[name].windows for p in parts]).sum()) for name in parts[0]}
    return {f"{name}/{k}": v for name, values in parts_as_floats(epoch).items() for k, v in values.items()}


# ---- step 2 ------------------------------------------------------------------------------------

@dataclass
class DiaryResult:
    best_epoch: int                       # the epoch kept (the last, without validation)
    best_score: float                     # its select_by score (NaN without validation)
    history: List[dict] = field(default_factory=list)


def class_counts(recordings: Dict[str, Recording]) -> Dict[str, np.ndarray]:
    """The answered hours per class of each target ("now" labels), plus 1 each so no class is impossible.

    Hours count by their weight (Recording.weights), so the class balance is the reweighted data's.
    """
    counts = {t: np.ones(k) for t, k in TARGET_CLASSES.items()}
    for rec in recordings.values():
        for t in counts:
            label = f"{t}_now"
            if label not in rec.labels:
                continue
            column = np.array([row[rec.labels.index(label)] for row in rec.answers.values()])
            weight = np.array([1.0 if rec.weights is None else rec.weights.get(h, 1.0) for h in rec.answers])
            known = ~np.isnan(column)
            counts[t] += np.bincount(column[known].astype(int), weights=weight[known],
                                     minlength=len(counts[t]))[:len(counts[t])]
    return counts


def train_diary(net: MotorNetwork, train: Dict[str, Recording], val: Optional[Dict[str, Recording]],
                settings: DiarySettings, device, rng: np.random.Generator,
                epochs: Optional[int] = None) -> DiaryResult:
    """Train the heads and pooling on the diary (the network must hold its pretrained parts).

    Args:
        val: Validation patients for early stopping, or None for the final fit.
        epochs: Without validation, how many epochs (from cross-validation); default max_epochs.

    Leaves the network with the kept epoch's weights.
    """
    if settings.feature_noise and not isinstance(net.movement, Harnet10Projection):
        raise ConfigError("feature_noise is in harnet10's feature SDs; this encoder has none")
    counts = class_counts(train)
    net.heads.set_priors(counts)
    net.loss.set_class_weights({t: balanced_weights(c, settings.class_weight_power, settings.class_weight_cap)
                                for t, c in counts.items()})
    net.freeze_all_but(settings.learn, keep_dropout=settings.keep_dropout)
    net.to(device)
    opt = torch.optim.AdamW(net.parameter_groups(settings.weight_decay), lr=settings.lr)
    learning = [p for p in net.parameters() if p.requires_grad]
    pin, amp = _device_type(device) == "cuda", settings.amp and _device_type(device) == "cuda"
    n_epochs = settings.max_epochs if val else (epochs or settings.max_epochs)
    result, best_state = DiaryResult(0, float("nan")), None
    for epoch in range(1, n_epochs + 1):
        net.train()

        def epoch_batches():
            for group in batches(tile_chunks(train, settings.chunk_h, rng), settings.batch, rng):
                examples = [make_example(train[pid], start, settings.chunk_h, settings.past_h, 0.0,
                                         settings.check_ins_per_day, rng) for pid, start in group]
                examples = [e for e in examples if e.has_graded_windows]
                if examples:
                    yield collate(examples, net.labels, pin=pin)

        losses = []
        for c in prefetch(epoch_batches()):
            batch = c.batch.to(device, non_blocking=pin)
            if settings.feature_noise:
                # feature_noise SDs of each raw feature: the encoder standardises by 1 / scale.
                noise = torch.randn_like(batch.movement) * (settings.feature_noise / net.movement.scale)
                batch = _replace_movement(batch, batch.movement + noise)
            with torch.autocast(_device_type(device), dtype=torch.bfloat16, enabled=amp):
                out = net(batch)
            opt.zero_grad(set_to_none=True)
            out.loss.total.backward()
            torch.nn.utils.clip_grad_norm_(learning, settings.grad_clip)
            opt.step()
            losses.append(out.loss.total.detach())
        line = {"epoch": epoch, "train_loss": torch.stack(losses).mean().item() if losses else float("nan")}
        if val:
            rows = score_hours(net, val, settings.eval_past_h, device, settings.eval_batch, amp=settings.amp)
            line.update(within_scores(rows))
            score = line[settings.select_by]
            if math.isfinite(score) and not score <= result.best_score:     # NaN best: anything finite wins
                result.best_epoch, result.best_score = epoch, score
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in net.state_dict().items()}
        result.history.append(line)
        logger.info("diary epoch %d: %s", epoch, _describe(line), extra={"event": "diary_epoch", **line})
        if val and result.best_epoch and epoch - result.best_epoch >= settings.patience:
            logger.info("stopping: no better epoch for %d", settings.patience,
                        extra={"event": "early_stop", "best_epoch": result.best_epoch})
            break
    if val and best_state is None:
        logger.warning("no epoch gave a finite validation score; keeping the last",
                       extra={"event": "no_valid_score"})
        result.best_epoch = len(result.history)
    elif best_state is not None:
        net.load_state_dict(best_state)
    else:
        result.best_epoch = len(result.history)
    return result


def _replace_movement(batch, movement):
    return replace(batch, movement=movement)


# ---- scoring hours -----------------------------------------------------------------------------

@torch.inference_mode()
def score_hours(net: MotorNetwork, recordings: Dict[str, Recording], past_hours: Sequence[int], device,
                batch_size: int = 16, amp: bool = True, check_ins_per_day: float = 0.0,
                rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
    """Every answered hour scored on its own, with exactly L hours of past in front, for each L.

    Returns:
        One row per (participant, hour, past_h) with windows: the answers (<label>_y) and the
        pooled probabilities (<label>_p, an array per row).
    """
    net.to(device).eval()
    pin, amp = _device_type(device) == "cuda", amp and _device_type(device) == "cuda"
    rows = []

    def jobs():
        for past in past_hours:
            items = [(pid, h) for pid, rec in recordings.items() for h in answered_hours(rec)]
            for k in range(0, len(items), batch_size):
                group = items[k:k + batch_size]
                examples = [make_example(recordings[pid], h, 1, past, 0.0, check_ins_per_day, rng)
                            for pid, h in group]
                keep = [i for i, e in enumerate(examples) if e.has_graded_windows]
                if keep:
                    yield past, [group[i] for i in keep], collate([examples[i] for i in keep], net.labels, pin=pin)

    for past, group, c in prefetch(jobs()):
        batch = c.batch.to(device, non_blocking=pin)
        with torch.autocast(_device_type(device), dtype=torch.bfloat16, enabled=amp):
            out = net(batch)
        probs = {label: out.pooled[label].attention[:, 0].float().cpu().numpy() for label in net.labels}
        has = out.has_windows[:, 0].cpu().numpy()
        for i, (pid, hour) in enumerate(group):
            if not has[i]:
                continue
            row = {"participant_id": pid, "hour_index": hour, "past_h": past}
            for label in net.labels:
                row[f"{label}_y"] = recordings[pid].answer(label, hour)
                row[f"{label}_p"] = probs[label][i]
            rows.append(row)
    return pd.DataFrame(rows)


def within_scores(rows: pd.DataFrame, horizon: str = "now") -> Dict[str, float]:
    """The within-patient AUCs, each averaged over the past lengths, and their mean (within_auc)."""
    if rows.empty:
        return {"off_auc_within": math.nan, "dyskinetic_auc_within": math.nan,
                "tremor_any_auc_within": math.nan, "within_auc": math.nan}
    per_past = []
    for _, r in rows.groupby("past_h"):
        state = r[r[f"state_{horizon}_y"].notna()]
        tremor = r[r[f"tremor_{horizon}_y"].notna()]
        p_state = np.stack(state[f"state_{horizon}_p"]) if len(state) else np.zeros((0, 7))
        p_tremor = np.stack(tremor[f"tremor_{horizon}_p"]) if len(tremor) else np.zeros((0, 3))
        y_state, y_tremor = state[f"state_{horizon}_y"].to_numpy(int), tremor[f"tremor_{horizon}_y"].to_numpy(int)
        off, dys = list(STATE_GROUPS["off"]), list(STATE_GROUPS["dyskinetic"])
        per_past.append({
            "off_auc_within": _auc_within(state["participant_id"], np.isin(y_state, off), p_state[:, off].sum(1)),
            "dyskinetic_auc_within": _auc_within(state["participant_id"], np.isin(y_state, dys),
                                                 p_state[:, dys].sum(1)),
            "tremor_any_auc_within": _auc_within(tremor["participant_id"], y_tremor >= 1, p_tremor[:, 1:].sum(1)),
        })
    out = {k: float(np.nanmean([m[k] for m in per_past])) if any(np.isfinite(m[k]) for m in per_past)
           else math.nan for k in per_past[0]}
    finite = [v for v in out.values() if math.isfinite(v)]
    out["within_auc"] = float(np.mean(finite)) if finite else math.nan
    return out


def _auc_within(pids, y, p):
    """The mean of each patient's own AUC, over patients with both classes (NaN if none has)."""
    frame = pd.DataFrame({"pid": np.asarray(pids), "y": np.asarray(y, bool), "p": np.asarray(p, float)})
    aucs = [roc_auc_score(g.y, g.p) for _, g in frame.groupby("pid") if 0 < g.y.sum() < len(g)]
    return float(np.mean(aucs)) if aucs else math.nan


# ---- helpers -----------------------------------------------------------------------------------

def prefetch(items: Iterable, depth: int = 2) -> Iterator:
    """items, built in a background thread up to depth ahead, so building overlaps the GPU's work.

    The thread does everything the iterable does (including its random draws), so the order and
    the results are the same as without it. An error in the thread is raised here.
    """
    q: "queue.Queue" = queue.Queue(maxsize=depth)
    done, stop = object(), threading.Event()

    def put(item):
        """Put item unless the consumer has left (stop set): never blocks forever on a full queue."""
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def work():
        try:
            for item in items:
                if not put(item):
                    return
            put(done)
        except BaseException as e:            # handed to the consumer, raised there
            put(e)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    try:
        while True:
            item = q.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        thread.join(timeout=5)


def _device_type(device) -> str:
    return torch.device(device).type


def _describe(line: dict) -> str:
    return ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in line.items()
                     if k != "epoch")


__all__ = ["DiaryResult", "DiarySettings", "PretrainSettings", "class_counts", "prefetch", "pretrain",
           "score_hours", "train_diary", "within_scores"]
