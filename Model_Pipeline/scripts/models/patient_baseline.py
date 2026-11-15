"""Each patient's own normal: harnet10 features relative to that patient's recent past.

Owner: P2.

Why. Trained across patients, a model learns who is usually OFF instead of when someone is OFF:
OFF ranges from 1% to 98% of a patient's waking hours, and a few patients who are nearly always
OFF give most of the OFF hours. Measured on the development patients (3 folds, logistic on the
features): within-patient OFF AUC 0.52 as they are, 0.59 to 0.63 relative to each patient's own
mean and SD. The window model sees at most 8 hours of past, too little to learn a patient's normal.

How (mode "rolling"). For each diary hour h, the patient's normal is the mean and SD of each
feature over their worn windows in the window_h hours before h (h itself left out), so it uses
only the past, as the app can live. It is shrunk towards the cohort's mean and SD (the training
patients' features) with the weight of warmup_h hours of windows, so a patient's first hours lean
on the cohort and their own normal takes over as their data builds up:

    mean_h = (sum of x + k * cohort_mean) / (n + k)                k = warmup_h x windows per hour
    var_h  = (sum of x^2 + k * (cohort_sd^2 + cohort_mean^2)) / (n + k) - mean_h^2
    x'     = (x - mean_h) / sd_h

A feature with no spread (sd below MIN_SD) is only centred, as the movement encoder does.

    settings = check_settings(params["baseline"])                 # {"mode", "window_h", "warmup_h"}
    cohort = feature_stats(chunks)                                # (mean, sd) over the training windows
    baseline = rolling_baseline(time_s, features, cohort, settings, window_s=10)
    x = baseline.apply(time_s[rows], features[rows])

Every model family that reads harnet10 features uses this one function, at fit and at predict,
so training and serving compute a patient's normal the same way. The cohort mean and SD are saved
with the model. A live service keeps each user's last window_h hours of features (or the running
sums per hour) and calls the same function.
"""

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Tuple

import numpy as np

from scripts.core.errors import ConfigError, ModelError

HOUR = 3600
MODES = ("none", "rolling")
DEFAULTS = {"mode": "none", "window_h": 168, "warmup_h": 24}
MIN_SD = 1e-6
BLOCK = 16_384           # windows read at a time from a memory-mapped cache file


def check_settings(settings: Optional[Mapping]) -> dict:
    """The baseline settings, defaults filled in.

    Raises:
        ConfigError: an unknown key, an unknown mode, or a window or warm-up that isn't a positive
            whole number of hours (the warm-up may be 0: no shrinking towards the cohort).
    """
    settings = dict(settings or {})
    unknown = set(settings) - set(DEFAULTS)
    if unknown:
        raise ConfigError(f"baseline takes {sorted(DEFAULTS)}, not {sorted(unknown)}")
    out = {**DEFAULTS, **settings}
    if out["mode"] not in MODES:
        raise ConfigError(f"baseline.mode must be one of {MODES}, not {out['mode']!r}")
    for name, low in (("window_h", 1), ("warmup_h", 0)):
        v = out[name]
        if isinstance(v, bool) or not isinstance(v, int) or v < low:
            raise ConfigError(f"baseline.{name} must be a whole number of hours of {low} or more, not {v!r}")
    return out


def feature_stats(chunks: Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """(mean, sd) per feature over every row of every chunk, combined exactly (Chan et al.), in float64."""
    n, mean, m2 = 0, None, None
    for chunk in chunks:
        for start in range(0, len(chunk), BLOCK):
            block = np.asarray(chunk[start:start + BLOCK], dtype=np.float64)
            if not len(block):
                continue
            if mean is None:
                mean, m2 = np.zeros(block.shape[1]), np.zeros(block.shape[1])
            nb, bm = len(block), block.mean(axis=0)
            delta = bm - mean
            m2 += np.square(block - bm).sum(axis=0) + np.square(delta) * (n * nb / (n + nb))
            mean += delta * (nb / (n + nb))
            n += nb
    if n < 2:
        raise ModelError(f"the cohort baseline needs at least 2 windows, not {n}")
    return mean, np.sqrt(m2 / (n - 1))


@dataclass(frozen=True, eq=False)
class Baseline:
    """One patient's normal for every diary hour from first_hour: mean and 1 / sd per feature."""

    first_hour: int
    mean: np.ndarray        # (hours, features) float32
    inv_sd: np.ndarray      # (hours, features) float32

    def apply(self, time_s: np.ndarray, features: np.ndarray) -> np.ndarray:
        """(rows, features) -> the same rows relative to the patient's normal of each row's hour, float32."""
        rows = np.clip((np.asarray(time_s) // HOUR).astype(np.int64) - self.first_hour, 0, len(self.mean) - 1)
        x = np.asarray(features, dtype=np.float32)
        return (x - self.mean[rows]) * self.inv_sd[rows]


def rolling_baseline(time_s: np.ndarray, features: np.ndarray, cohort: Tuple[np.ndarray, np.ndarray],
                     settings: Mapping, window_s: float) -> Baseline:
    """A patient's normal per hour from their own past windows (see the top of this file).

    Args:
        time_s: (n,) each worn window's start on the diary clock, increasing.
        features: (n, F) their features (a memory-mapped cache file is read a block at a time).
        cohort: (mean, sd) per feature over the training windows, from feature_stats.
        settings: check_settings's output.
        window_s: The window length, for windows per hour.
    """
    time_s = np.asarray(time_s, dtype=np.float64)
    if len(time_s) != len(features):
        raise ModelError(f"{len(features)} feature rows for {len(time_s)} windows")
    cohort_mean, cohort_sd = (np.asarray(v, dtype=np.float64).reshape(-1) for v in cohort)
    n_features = cohort_mean.shape[0]
    if not len(time_s):
        return Baseline(0, cohort_mean[None].astype(np.float32),
                        _inverse(cohort_sd)[None].astype(np.float32))
    hour = (time_s // HOUR).astype(np.int64)
    first, last = int(hour[0]), int(hour[-1])
    n_hours = last - first + 1
    count = np.bincount(hour - first, minlength=n_hours).astype(np.float64)
    s1, s2 = np.zeros((n_hours, n_features)), np.zeros((n_hours, n_features))
    for start in range(0, len(time_s), BLOCK):
        block = np.asarray(features[start:start + BLOCK], dtype=np.float64)
        rows = hour[start:start + BLOCK] - first
        np.add.at(s1, rows, block)
        np.add.at(s2, rows, np.square(block))
    # Sums over hours [h - window_h, h): cumulative sums with a leading zero, so hour h itself is out.
    c0 = np.concatenate([[0.0], np.cumsum(count)])
    c1 = np.vstack([np.zeros((1, n_features)), np.cumsum(s1, axis=0)])
    c2 = np.vstack([np.zeros((1, n_features)), np.cumsum(s2, axis=0)])
    end = np.arange(n_hours)
    begin = np.maximum(end - settings["window_h"], 0)
    n = (c0[end] - c0[begin])[:, None]
    sum1, sum2 = c1[end] - c1[begin], c2[end] - c2[begin]
    k = settings["warmup_h"] * HOUR / window_s
    denom = np.maximum(n + k, 1.0)
    mean = (sum1 + k * cohort_mean) / denom
    var = (sum2 + k * (np.square(cohort_sd) + np.square(cohort_mean))) / denom - np.square(mean)
    # No past and no warm-up: the cohort's own normal.
    empty = (n[:, 0] + k) <= 0
    mean[empty], var[empty] = cohort_mean, np.square(cohort_sd)
    sd = np.sqrt(np.maximum(var, 0.0))
    return Baseline(first, mean.astype(np.float32), _inverse(sd).astype(np.float32))


def _inverse(sd):
    return np.where(sd > MIN_SD, 1.0 / np.maximum(sd, MIN_SD), 1.0)


class Normalised:
    """A patient's features read relative to their normal, a slice or a set of rows at a time.

    Behaves like the (n, F) array for len, shape and indexing, so code that reads features in
    blocks (features[start:stop]) or picks rows (features[rows]) reads normalised ones.
    """

    def __init__(self, features: np.ndarray, time_s: np.ndarray, baseline: Baseline):
        self.features, self.time_s, self.baseline = features, np.asarray(time_s), baseline
        self.shape = features.shape

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.baseline.apply(self.time_s[index], self.features[index])


__all__ = ["DEFAULTS", "MODES", "Baseline", "Normalised", "check_settings", "feature_stats", "rolling_baseline"]
