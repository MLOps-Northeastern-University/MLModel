"""MotorCausalTransformer: the motor causal transformer behind the core/model.py contract.

The only file of this package the pipeline imports. The train, tune, evaluate, bias and register
steps see an ordinary Model:

    model = MotorCausalTransformer(targets=["sleep", "state", "tremor"], spec=spec)
    model.fit(train, val=val, seed=0)       # step 1 pretraining, then step 2 the diary step
    predictions = model.predict(data)       # every diary hour, and every worn window
    model.save(folder); MotorCausalTransformer.load(folder)

fit, on the training participants only. sample_weight (one per training hour, for bias mitigation
by reweighting) weights each hour in the diary step's loss and class balance; pretraining learns no
labels, so it reads every hour's movement as it is:
    1. harnet10's features for every worn window, computed once and cached (harnet.py)
    2. the movement encoder's standardisation, from the training windows
    3. pretraining: predict the movement ahead (training.pretrain)
    4. the diary step: heads and pooling, with val for early stopping, or a fixed number of
       epochs (params "final_epochs", from cross-validation) without it (training.train_diary)

Step 3 is most of a fit's time and doesn't depend on the diary step's settings, so its result is
cached (pretrained_cache.py): a fit with the same pretraining (same patients, data, settings, seed,
code and device) loads the body instead, and goes on exactly as if it had pretrained. That makes
tuning the diary step's settings, and rerunning train, about 2 to 3 times faster.

predict scores each diary hour with exactly score_past_h hours of past in front of it (8 h,
the longest validation tested). An hour's windows get their probabilities from that pass, and
the hour gets its pooled prediction. An hour with no worn windows has nothing to score, so it
gets the training hours' class mix, and the count is logged.

Settings (params), each block passed to its part:
    network       MotorNetwork's settings (network.py)
    pretrain      PretrainSettings (training.py); epochs 0 skips pretraining
    diary         DiarySettings (training.py)
    dose_span_s   how long a dose stays marked (1,800 s)
    score_past_h  hours of past each scored hour sees (8)
    final_epochs  diary epochs without validation (None: diary.max_epochs)
    baseline      each patient's features relative to their own past (models/patient_baseline.py):
                  {"mode": "none" | "rolling", "window_h": 168, "warmup_h": 24}; the cohort mean and
                  SD it shrinks towards come from the training windows and are saved with the model
A block given in params is merged into its defaults, so {"diary": {"lr": 1e-3}} changes only lr.

Where things are, which isn't part of a model (environment variables):
    MODEL_HARNET_HUB     the local copy of OxWearables/ssl-wearables (~/.cache/torch/hub/...)
    MODEL_FEATURE_CACHE  where harnet10's features are cached (artifacts/cache/harnet10)
    MODEL_DEVICE         cuda or cpu (cuda when there is one)
    MODEL_PRETRAIN_CACHE where pretrained bodies are cached (artifacts/cache/pretrained), "off" for never
A saved model records harnet10's weights fingerprint, and predict refuses other weights.
"""

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from scripts.core.errors import ConfigError, ModelError
from scripts.core.model import Model, Predictions, WindowPredictions
from scripts.core.targets import HOUR_INDEX, PARTICIPANT, get_target, parse_label
from scripts.models.motor_causal_transformer.data import collate, make_example, prepare
from scripts.models.motor_causal_transformer.harnet import check_spec, default_device, feature_cache
from scripts.models import patient_baseline
from scripts.models.motor_causal_transformer.heads import probabilities
from scripts.models.motor_causal_transformer.network import MotorNetwork
from scripts.models.motor_causal_transformer.pretrained_cache import (
    PretrainedCache, random_states, restore_random_states)
from scripts.models.motor_causal_transformer.pretraining import MovementForecast
from scripts.models.motor_causal_transformer.training import (
    PRETRAINED, DiarySettings, PretrainSettings, class_counts, prefetch, pretrain, train_diary)
from scripts.utils.logging_setup import get_logger

logger = get_logger(__name__)

STATE_FILE, INFO_FILE = "network.pt", "training.json"
_device = default_device


class MotorCausalTransformer(Model):
    """The motor causal transformer as a model family (see the top of this file)."""

    family = "motor_causal_transformer"
    needs = ("windows", "events", "patients")
    scores_windows = True

    @classmethod
    def default_params(cls):
        return {"network": {}, "pretrain": asdict(PretrainSettings()), "diary": _jsonable(asdict(DiarySettings())),
                "dose_span_s": 1800.0, "score_past_h": 8, "final_epochs": None,
                "baseline": dict(patient_baseline.DEFAULTS)}

    def _merged_params(self, params):
        """As Model's, except that a block (a dict) given in params is merged into its defaults."""
        defaults, params = self.default_params(), copy.deepcopy(dict(params or {}))
        for key, value in params.items():
            if isinstance(defaults.get(key), dict) and isinstance(value, dict):
                params[key] = {**defaults[key], **value}
        return super()._merged_params(params)

    def __init__(self, targets, horizons=("now", "next_hour"), params=None, spec=None):
        super().__init__(targets, horizons, params, spec)
        p = self.params
        self._pretrain_settings = _settings(PretrainSettings, p["pretrain"], "pretrain")
        self._diary_settings = _settings(DiarySettings, p["diary"], "diary")
        for name in ("score_past_h",):
            if isinstance(p[name], bool) or not isinstance(p[name], int) or p[name] < 0:
                raise ConfigError(f"{name} must be whole hours of 0 or more, not {p[name]!r}")
        if p["final_epochs"] is not None and (isinstance(p["final_epochs"], bool)
                                              or not isinstance(p["final_epochs"], int) or p["final_epochs"] < 1):
            raise ConfigError(f"final_epochs must be a positive whole number or None, not {p['final_epochs']!r}")
        self._baseline_settings = patient_baseline.check_settings(p["baseline"])
        network = copy.deepcopy(p["network"])
        heads = network.setdefault("heads", {})
        if "horizons" in heads and tuple(heads["horizons"]) != self.horizons:
            raise ConfigError("network.heads.horizons is set from the model's horizons; don't set it")
        heads["horizons"] = list(self.horizons)
        self._network_settings = network
        self._net = self._build_network()                   # also checks the network settings now
        self._fingerprint = None                            # harnet10's weights, once fitted
        self._prior = None                                  # {label: class mix}, for hours with no windows
        self._result = None
        self._cohort = None                                 # (mean, sd) the patient baselines shrink towards

    def _build_network(self):
        try:
            return MotorNetwork(**copy.deepcopy(self._network_settings))
        except TypeError as e:
            raise ConfigError(f"network settings: {e}") from None

    # ---- fit -----------------------------------------------------------------------------------

    def _fit(self, train, val, sample_weight, seed):
        check_spec(self.spec)
        device = _device()
        cache = self._cache(device)
        # Seeded after harnet10 is ready: loading it builds a network, which draws from PyTorch's
        # random numbers, so seeding first would make the starting weights depend on whether it
        # was loaded in this process or already there.
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = self._build_network()
        self._cohort = None
        train_recs = self._recordings(train, cache, sample_weight)
        if self._baseline_settings["mode"] != "none":
            # The cohort's normal, from the training windows only, before any recording is normalised.
            self._cohort = patient_baseline.feature_stats(r.features for r in train_recs.values())
            train_recs = self._with_baselines(train_recs)
        val_recs = self._recordings(val, cache) if val is not None else None
        logger.info("training %s on %d patients (%d worn windows)", self.family, len(train_recs),
                    sum(len(r.time_s) for r in train_recs.values()),
                    extra={"event": "mct_fit", "train": len(train_recs), "val": len(val_recs or {})})
        net.movement.fit(_feature_blocks(train_recs.values()))
        history = {"pretrain": [], "diary": []}
        if self._pretrain_settings.epochs:
            history["pretrain"] = self._pretrained(net, train_recs, val_recs, cache, device, rng, seed)
        result = train_diary(net, train_recs, val_recs, self._diary_settings, device, rng,
                             epochs=self.params["final_epochs"])
        history["diary"] = result.history
        self._net = net.to("cpu").eval()
        counts = class_counts(train_recs)
        self._prior = {label: (counts[parse_label(label)[0]] / counts[parse_label(label)[0]].sum()).tolist()
                       for label in self.labels}
        self._result = {"best_epoch": result.best_epoch, "best_score": result.best_score, "history": history}
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    def _pretrained(self, net, train_recs, val_recs, cache, device, rng, seed):
        """Pretrain net in place, or load the very same pretraining from the cache. Returns its history."""
        store = PretrainedCache.from_environment()
        if store is not None:
            material = {"network": self._network_settings, "pretrain": asdict(self._pretrain_settings),
                        "dose_span_s": self.params["dose_span_s"], "seed": seed, "labels": list(net.labels),
                        "baseline": self._baseline_settings,
                        "device": torch.device(device).type, "harnet10": cache.harnet.fingerprint}
            key = store.key(train_recs, val_recs, cache.keys, material)
            entry = store.load(key)
            if entry is not None:
                net.load_parts(entry["state"], PRETRAINED)
                restore_random_states(rng, entry["rng"])
                return entry["history"]
        forecast = MovementForecast(d_model=net.decoder.d_model, feature_dim=net.movement.input_shape[0])
        history = pretrain(net, forecast, train_recs, self._pretrain_settings, device, rng, val=val_recs)
        del forecast
        if store is not None:
            state = {k: v for k, v in net.state_dict().items() if k.split(".", 1)[0] in PRETRAINED}
            store.save(key, state, history, random_states(rng), about={
                "family": self.family, "seed": seed, "train_patients": len(train_recs),
                "val_patients": len(val_recs or {}), "pretrain": asdict(self._pretrain_settings)})
        return history

    @property
    def training_result(self):
        """The fit's best epoch, its validation score and every epoch's log (None before fitting)."""
        return copy.deepcopy(self._result)

    def _cache(self, device):
        cache = feature_cache(device, self._fingerprint)
        self._fingerprint = cache.harnet.fingerprint
        return cache

    def _recordings(self, data, cache, sample_weight=None):
        # Only the model's own labels are graded: the others are left out (NaN) of the answers.
        mine = data.labels.reindex(columns=[c for c in data.labels.columns if c in self.labels])
        data = _with_labels(data, mine)
        recordings = prepare(data, cache, self._net.labels, window_s=self.spec.seconds,
                             dose_span_s=self.params["dose_span_s"], sample_weight=sample_weight)
        return self._with_baselines(recordings) if self._cohort is not None else recordings

    def _with_baselines(self, recordings):
        """Each recording with its patient's rolling baseline (models/patient_baseline.py)."""
        return {pid: replace(rec, baseline=patient_baseline.rolling_baseline(
                    rec.time_s, rec.features, self._cohort, self._baseline_settings, self.spec.seconds))
                for pid, rec in recordings.items()}

    # ---- predict -------------------------------------------------------------------------------

    @torch.inference_mode()
    def _predict(self, data):
        device = _device()
        recordings = self._recordings(data, self._cache(device))
        net = self._net.to(device).eval()
        past = self.params["score_past_h"]
        amp = self._diary_settings.amp and device.startswith("cuda")
        hourly, windows = {}, {}

        def jobs():
            for pid, rec in recordings.items():
                if not len(rec.time_s):
                    continue
                first, last = rec.hours()
                hours = list(range(first, last + 1))
                for k in range(0, len(hours), self._diary_settings.eval_batch):
                    examples = [make_example(rec, h, 1, past) for h in hours[k:k + self._diary_settings.eval_batch]]
                    examples = [e for e in examples if e.has_graded_windows]
                    if examples:
                        yield pid, examples, collate(examples, net.labels)

        for pid, examples, c in prefetch(jobs()):
            batch = c.batch.to(device)
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=amp):
                out = net(batch)
            window_p = {label: p.float().cpu().numpy() for label, p in probabilities(out.logits).items()}
            pooled = {label: out.pooled[label].attention[:, 0].float().cpu().numpy() for label in self.labels}
            graded = c.graded.numpy()
            for i, e in enumerate(examples):
                hour = int(e.hours[0])
                hourly[(pid, hour)] = {label: pooled[label][i] for label in self.labels}
                windows.setdefault(pid, []).append((e.time_s[:e.n_input][graded[i, :e.n_input]],
                                                    {label: window_p[label][i, :e.n_input][graded[i, :e.n_input]]
                                                     for label in self.labels}))
        self._net.to("cpu")
        return self._predictions(data, recordings, hourly, windows)

    def _predictions(self, data, recordings, hourly, windows):
        keys = list(zip(data.hours[PARTICIPANT], data.hours[HOUR_INDEX].astype(int)))
        missing = sum(k not in hourly for k in keys)
        if missing:
            logger.info("%d of %d hours have no worn windows: given the training class mix", missing, len(keys),
                        extra={"event": "hours_without_windows", "hours": missing})
        probs = {label: np.stack([hourly[k][label] if k in hourly else np.asarray(self._prior[label])
                                  for k in keys]) if keys else
                 np.zeros((0, len(get_target(parse_label(label)[0]).classes))) for label in self.labels}
        probs = {label: p / p.sum(axis=1, keepdims=True) if len(p) else p for label, p in probs.items()}
        window_predictions = {}
        for pid in data.participants:
            parts = sorted(windows.get(pid, []), key=lambda x: x[0][0] if len(x[0]) else 0)
            time_s = np.concatenate([t for t, _ in parts]) if parts else np.zeros(0)
            window_predictions[pid] = WindowPredictions(time_s, {
                label: (np.concatenate([p[label] for _, p in parts]) if parts else
                        np.zeros((0, len(get_target(parse_label(label)[0]).classes))))
                for label in self.labels})
            window_predictions[pid] = _renormalised(window_predictions[pid])
            expected = recordings[pid].time_s
            if not np.array_equal(window_predictions[pid].time_s, expected):
                raise ModelError(f"{self.family}: scored {len(time_s)} of {pid}'s {len(expected)} worn windows")
        return Predictions(data.hours, probs, window_predictions)

    # ---- save and load -------------------------------------------------------------------------

    def _save_weights(self, folder):
        torch.save(self._net.state_dict(), Path(folder) / STATE_FILE)
        (Path(folder) / INFO_FILE).write_text(json.dumps(
            {"harnet10": self._fingerprint, "prior": self._prior, "result": self._result,
             "cohort": None if self._cohort is None else [v.tolist() for v in self._cohort]},
            indent=1, default=float),
            encoding="utf-8")

    def _load_weights(self, folder):
        state = torch.load(Path(folder) / STATE_FILE, map_location="cpu", weights_only=True)
        self._net.load_state_dict(state)
        self._net.eval()
        info = json.loads((Path(folder) / INFO_FILE).read_text(encoding="utf-8"))
        self._fingerprint, self._prior, self._result = info["harnet10"], info["prior"], info["result"]
        cohort = info.get("cohort")
        if (cohort is None) != (self._baseline_settings["mode"] == "none"):
            state = "off" if cohort is None else "on"
            raise ModelError(f"{folder}: saved with the patient baseline {state}, but the settings say mode "
                             f"{self._baseline_settings['mode']!r}")
        self._cohort = None if cohort is None else tuple(np.asarray(v, dtype=np.float64) for v in cohort)


def _feature_blocks(recordings, block=patient_baseline.BLOCK):
    """Every recording's features as the model reads them (normalised if it has a baseline), a block at a time."""
    for rec in recordings:
        for start in range(0, len(rec.time_s), block):
            yield rec.window_features(start, start + block)


def _settings(cls, values, name):
    try:
        return cls(**values)
    except TypeError as e:
        raise ConfigError(f"{name} settings: {e}") from None


def _jsonable(d):
    return {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}


def _with_labels(data, labels):
    return replace(data, labels=labels)


def _renormalised(w):
    """Probabilities summed in float32 can be a hair off 1; the contract checks to 1e-6."""
    return WindowPredictions(w.time_s, {k: (p / p.sum(axis=1, keepdims=True)) if len(p) else p
                                        for k, p in w.probs.items()})


__all__ = ["MotorCausalTransformer"]
