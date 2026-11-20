"""What train and tune share, so a tuning trial and a training run are scored exactly the same way.

Owner: P2.

    data, fingerprint = load_development(cfg, split, models)       # the split's development patients only
    result = cross_validate(name, template, dev, split, seeds, tracker)
    result.cv, result.metrics()                                     # {label: {metric}}, and as tracker names
    model = fresh(template, {"diary": {"lr": 1e-3}})                # a new model, settings deep-merged

cross_validate fits a fresh copy of the candidate on each fold's training patients (its validation
patients are passed for early stopping), predicts the validation patients and scores them
(core/metrics.py), for every seed. Each fold is a tracker run, seed-<s>-fold-<k>, with its val/ scores,
fit and predict seconds, and the fit's epoch curves (run/<phase>_<value>, step = epoch). The cv scores
are each seed's out-of-fold scores, averaged over seeds; _fold_sd says how far one fold's score can be
from another's.

Scores count only hours with a label and at least one worn window: an hour without worn windows gets
the training class mix, which says nothing about the model. The rows (frames) keep every hour, with
a worn column, for the predictions files.

The tuned-settings file (artifacts/tuned_params.json) is how tune hands its result to train: per
candidate, the settings tuning chose, and what they were tuned on (the split, the data, the
candidate's own config). read_tuned reuses an entry tuned on another split or other data, with a
warning: the last tuned settings are kept until a run asks to tune again (run_model_pipeline --tune).
An entry tuned on another config of the candidate is skipped, so a changed config is never
overwritten by settings tuned for the old one.
"""

import copy
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from model_pipeline.core.metrics import label_scores, mean_scores
from model_pipeline.core.targets import HOUR_INDEX, PARTICIPANT, get_target, parse_label
from model_pipeline.utils.file_io import write_json
from model_pipeline.utils.logging_setup import get_logger

logger = get_logger(__name__)

SECONDS_PER_HOUR = 3600
TUNED_FILE = "tuned_params.json"


# ---- data ---------------------------------------------------------------------------------------

def load_development(cfg, split, models, include_test=False):
    """The Dataset of the split's development patients (and test patients only if include_test), with
    every group the models need plus patients (for the no-DBS scores), and a fingerprint of the data.

    Raises:
        SchemaError: The data and the split don't name the same patients.
    """
    from model_pipeline.data.loaders import load_dataset
    paths = cfg["resolved_paths"]
    everyone = pd.read_csv(paths["hours_csv"], usecols=[PARTICIPANT])[PARTICIPANT].unique()
    split.check(everyone)
    groups = sorted({g for _, m in models for g in m.needs} | {"patients"})
    people = list(split.development) + (list(split.test) if include_test else [])
    return load_dataset(cfg, groups=groups, participants=people), data_fingerprint(paths)


TABLES = ("hours_csv", "events_csv", "patients_csv", "features_csv")
SAMPLE_BYTES = 1 << 20


def data_fingerprint(paths) -> str:
    """Names the data a run could read, whichever groups it loaded, so tune and train agree on it.

    Every input table in paths that exists, by content; the windows folder by its manifest, and each
    file by its name, size and the first and last MiB of its contents (reading every window would take
    minutes; a file reprocessed in place changes in those bytes, even at the same size).
    """
    digest = hashlib.blake2b(digest_size=8)
    for key in TABLES:
        path = paths.get(key)
        if path is None or not Path(path).exists():
            continue
        digest.update(key.encode("utf-8"))
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
    folder = Path(paths["windows_folder"]) if paths.get("windows_folder") else None
    if folder is not None and folder.exists():
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            size = path.stat().st_size
            digest.update(f"{path.relative_to(folder).as_posix()}:{size};".encode("utf-8"))
            with open(path, "rb") as f:
                digest.update(f.read(SAMPLE_BYTES))
                if size > 2 * SAMPLE_BYTES:
                    f.seek(-SAMPLE_BYTES, 2)
                    digest.update(f.read(SAMPLE_BYTES))
    return digest.hexdigest()


# ---- models -------------------------------------------------------------------------------------

def merge_params(base: Mapping, overrides: Optional[Mapping]) -> Dict:
    """base with overrides merged in, recursively: {"diary": {"lr": 1}} changes only diary.lr."""
    out = copy.deepcopy(dict(base))
    for key, value in (overrides or {}).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = merge_params(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def fresh(template, overrides=None):
    """A new, unfitted model like template, with some settings changed (deep-merged)."""
    return type(template)(template.targets, template.horizons, merge_params(template.params, overrides),
                          template.spec)


def candidate_fingerprint(template) -> str:
    """Names a candidate's configuration: family, targets, horizons and every setting."""
    text = json.dumps({"family": template.family, "targets": list(template.targets),
                       "horizons": list(template.horizons), "params": template.params}, sort_keys=True, default=str)
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


# ---- cross-validation ---------------------------------------------------------------------------

@dataclass
class CVResult:
    """One candidate's cross-validation.

    Attributes:
        cv: {label: {metric: value}}, each seed's out-of-fold scores averaged over seeds.
        fold_sd: {label: {metric: sd}} across folds and seeds.
        frames: {label: rows}, every seed's out-of-fold hours.
        best_epochs: Each fit's best epoch, for families that report one.
        pretraining: The folds' pretraining summaries averaged (run/pretrain_...), empty if none.
    """

    cv: Dict[str, Dict[str, float]]
    fold_sd: Dict[str, Dict[str, float]]
    frames: Dict[str, pd.DataFrame]
    best_epochs: List[int] = field(default_factory=list)
    pretraining: Dict[str, float] = field(default_factory=dict)

    def metrics(self) -> Dict[str, float]:
        """Everything as tracker names: cv/<label>/<metric>, its _fold_sd, and run/pretrain_..."""
        return {**named("cv", self.cv), **named("cv", self.fold_sd, suffix="_fold_sd"), **self.pretraining}


def cross_validate(name, template, dev, split, seeds: Sequence[int], tracker, charts=None) -> CVResult:
    """Fit, predict and score a fresh copy of template on every fold, for every seed (see the top of this file).

    charts: a reporting.writer.ChartWriter to give each fold's run its own charts (if its folds flag is
    on), or None for none (tuning trials).
    """
    fold_scores, seed_scores, frames, best_epochs, pretraining = {}, {}, [], [], []
    for seed in seeds:
        seed_frames = []
        for k, (train_ids, val_ids) in enumerate(split.pairs(), start=1):
            with tracker.run(f"seed-{seed}-fold-{k}", tags={"fold": str(k), "seed": str(seed)}):
                model = fresh(template)
                val = dev.subset(val_ids)
                t0 = time.perf_counter()
                model.fit(dev.subset(train_ids), val=val, seed=seed)
                t1 = time.perf_counter()
                fold_frames = rows(name, model.predict(val), val, fold=str(k), seed=seed)
                scores = score(fold_frames)
                tracker.log_metrics({**named("val", scores), "run/fit_s": t1 - t0,
                                     "run/predict_s": time.perf_counter() - t1})
                log_curves(tracker, model)
                if charts is not None and charts.folds:
                    charts.predictions(fold_frames, scores, f"{name}/seed-{seed}-fold-{k}",
                                       f"{name}, seed {seed}, fold {k} (validation patients)")
                best = (_training_result(model) or {}).get("best_epoch")
                if best is not None:
                    best_epochs.append(int(best))
                pretraining.append(pretraining_summary(model))
                fold_scores[(seed, k)] = scores
                seed_frames.append(fold_frames)
        merged = concat(seed_frames)
        seed_scores[seed] = score(merged)
        frames.append(merged)
    return CVResult(mean_scores(seed_scores), _fold_sd(fold_scores), concat(frames), best_epochs,
                    _mean_over(pretraining))


def rows(name, predictions, data, fold, seed) -> Dict[str, pd.DataFrame]:
    """{label: one row per hour}: who, which fold, the label, the probabilities, and if the hour was worn."""
    hours = data.hours.reset_index(drop=True)
    labels = data.labels.reset_index(drop=True)
    worn = _worn(data, hours)
    no_dbs = _no_dbs(data, hours)
    out = {}
    for label in predictions.labels:
        classes = get_target(parse_label(label)[0]).classes
        frame = pd.DataFrame({PARTICIPANT: hours[PARTICIPANT].to_numpy(), HOUR_INDEX: hours[HOUR_INDEX].to_numpy(),
                              "fold": fold, "seed": seed, "model": name, "worn": worn, "no_dbs": no_dbs,
                              "y_true": labels[label].to_numpy() if label in labels else np.nan})
        out[label] = pd.concat([frame, pd.DataFrame(predictions.probs[label], columns=list(classes))], axis=1)
    return out


def score(frames) -> Dict[str, Dict[str, float]]:
    """{label: metrics} on the hours with a label and a worn window."""
    out = {}
    for label, frame in frames.items():
        classes = list(get_target(parse_label(label)[0]).classes)
        f = frame[frame["worn"].to_numpy(bool)]
        no_dbs = f["no_dbs"].to_numpy().astype(bool) if f["no_dbs"].notna().all() else None
        out[label] = label_scores(label, f["y_true"].to_numpy(float), f[classes].to_numpy(float),
                                  f[PARTICIPANT].to_numpy(), no_dbs)
    return out


def concat(parts) -> Dict[str, pd.DataFrame]:
    out = {}
    for part in parts:
        for label, frame in part.items():
            out.setdefault(label, []).append(frame)
    return {label: pd.concat(frames, ignore_index=True) for label, frames in out.items()}


def named(split, scores, suffix="") -> Dict[str, float]:
    """{label: {metric: v}} as tracker names, {split}/{label}/{metric}{suffix}, leaving out NaN."""
    return {f"{split}/{label}/{metric}{suffix}": v for label, s in scores.items() for metric, v in s.items()
            if isinstance(v, (int, float)) and math.isfinite(v)}


def write_predictions(folder, frames) -> None:
    """artifacts/predictions/<label>.csv, each written through a temporary file."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for label, parts in frames.items():
        table = pd.concat(parts, ignore_index=True).drop(columns=["no_dbs"])
        tmp = folder / f".{label}.csv.tmp"
        table.to_csv(tmp, index=False)
        tmp.replace(folder / f"{label}.csv")
    logger.info("predictions written for %d labels", len(frames),
                extra={"event": "predictions_written", "path": str(folder)})


# ---- what a fit reports -------------------------------------------------------------------------

def log_curves(tracker, model) -> None:
    """A fit's epoch-by-epoch history (if the family keeps one) as run/<phase>_<value>, step = epoch."""
    history = (_training_result(model) or {}).get("history")
    if not isinstance(history, dict):
        return
    for phase, lines in history.items():
        for line in lines or []:
            epoch = line.get("epoch")
            values = {f"run/{slug(phase)}_{slug(k)}": v for k, v in line.items()
                      if k != "epoch" and _finite(v)}
            if values and isinstance(epoch, int):
                tracker.log_metrics(values, step=epoch)


def pretraining_summary(model) -> Dict[str, float]:
    """A fit's pretraining, from its last epoch: each target's validation skill, their mean, and the time.

    Names match the fold's curves (run/pretrain_val_<target>_skill), so a candidate's values are the
    folds' last points averaged. Empty for a model that didn't pretrain.
    """
    history = (_training_result(model) or {}).get("history") or {}
    lines = history.get("pretrain") if isinstance(history, dict) else None
    if not lines:
        return {}
    skills = {f"run/pretrain_{slug(k)}": v for k, v in lines[-1].items()
              if k.startswith("val/") and k.endswith("/skill") and _finite(v)}
    out = dict(skills)
    if skills:
        out["run/pretrain_val_skill"] = float(np.mean(list(skills.values())))
    out["run/pretrain_seconds"] = float(sum(line.get("seconds", 0.0) for line in lines))
    return out


def slug(text) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(text).lower()).strip("_")


# ---- the tuned-settings file -------------------------------------------------------------------

def write_tuned(folder, entries: Mapping[str, Mapping]) -> Path:
    """Merge entries ({candidate: {...}}) into artifacts/tuned_params.json, keeping other candidates'."""
    path = Path(folder) / TUNED_FILE
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current.update(entries)
    return write_json(current, path)


def read_tuned(folder, models, split, fingerprint) -> Dict[str, Dict]:
    """{candidate: its entry} for the candidates in the file, checked against what they were tuned on.

    An entry holds "params" (the settings to merge over the candidate's config) and where they came
    from ("study", "best_trial", "value", "tuned_at"), plus "stale": what changed since it was tuned
    (["split"], ["data"], both, or empty).

    - Tuned on another split or other data: reused, with a warning. The last tuned settings are
      usually still the best guess, and tuning again is a choice (run_model_pipeline --tune).
    - Tuned on another config of the candidate: skipped, with a warning, so the candidate trains with
      its config. The old settings could undo the change (a tuned setting the new config sets too).
    - No file: nothing is tuned yet, so every candidate trains with its config, with a warning.
    """
    path = Path(folder) / TUNED_FILE
    if not path.exists():
        logger.warning("train.use_tuned is on, but there's no %s: every candidate trains with its config", path,
                       extra={"event": "tuned_missing"})
        return {}
    entries = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for name, template in models:
        entry = entries.get(name)
        if entry is None:
            continue
        if entry.get("candidate_hash") != candidate_fingerprint(template):
            logger.warning("%s's config changed since it was tuned: it trains with its config, not the tuned "
                           "settings (tune again with --tune)", name,
                           extra={"event": "tuned_skipped", "candidate": name})
            continue
        stale = [what for what, now in (("split", split.hash), ("data", fingerprint))
                 if entry.get(f"{what}_hash") != now]
        if stale:
            logger.warning("%s reuses settings tuned on a different %s (tuned %s; tune again with --tune)",
                           name, " and ".join(stale), entry.get("tuned_at"),
                           extra={"event": "tuned_stale", "candidate": name, "stale": stale})
        out[name] = {**entry, "stale": stale}
    return out


# ---- what every run records -------------------------------------------------------------------

def provenance():
    """Tags and params that say which code made a run: the git commit (and if it had uncommitted
    changes), Python and the library versions."""
    from model_pipeline.utils.logging_setup import environment
    env = environment(packages=("numpy", "pandas", "scikit-learn", "mlflow", "torch", "optuna"))
    tags = {"git_commit": env["git_commit"] or "unknown", "git_dirty": str(env["git_dirty"]).lower()}
    return tags, {"python": env["python"], "packages": env["packages"]}


# ---- helpers ------------------------------------------------------------------------------------

def _training_result(model):
    return getattr(model, "training_result", None)


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _worn(data, hours):
    """Per hour: does it have at least one worn window? All True for data without windows."""
    if data.windows is None:
        return np.ones(len(hours), bool)
    worn_hours = set()
    for pid in data.participants:
        w = data.windows[pid]
        for h in np.unique(np.floor(np.asarray(w.time_s)[np.asarray(w.worn)] / SECONDS_PER_HOUR).astype(np.int64)):
            worn_hours.add((pid, int(h)))
    return np.array([(p, int(h)) in worn_hours for p, h in zip(hours[PARTICIPANT], hours[HOUR_INDEX])], bool)


def _no_dbs(data, hours):
    """Per hour: is the patient without DBS? NaN when the patients table isn't loaded."""
    if data.patients is None:
        return np.full(len(hours), np.nan, dtype=object)
    dbs = dict(zip(data.patients[PARTICIPANT], data.patients["dbs"].astype(bool)))
    return np.array([not dbs[p] for p in hours[PARTICIPANT]], dtype=object)


def _fold_sd(fold_scores):
    """{label: {metric: sd across folds and seeds}}."""
    labels = {label for s in fold_scores.values() for label in s}
    out = {}
    for label in labels:
        metrics = {m for s in fold_scores.values() for m in s.get(label, {}) if m != "hours"}
        out[label] = {}
        for m in metrics:
            values = [s[label][m] for s in fold_scores.values() if math.isfinite(s.get(label, {}).get(m, math.nan))]
            out[label][m] = float(np.std(values)) if len(values) > 1 else math.nan
    return out


def _mean_over(parts):
    """{name: mean} of the names in parts (dicts), each over the parts that have it."""
    names = {n for part in parts for n in part}
    return {n: float(np.mean([part[n] for part in parts if n in part])) for n in sorted(names)}


__all__ = ["CVResult", "candidate_fingerprint", "concat", "cross_validate", "data_fingerprint", "fresh",
           "load_development", "log_curves", "merge_params", "named", "pretraining_summary", "provenance", "read_tuned",
           "rows", "score", "slug", "write_predictions", "write_tuned"]
