"""Tunes candidates' settings with Optuna, scoring every trial by the same cross-validation as train.

Owner: P2 (component D).

What to tune is config only (config/model.yaml, tune:), so a new model or a new setting needs no code
here: name the candidate (as in models.candidates) and the settings to search, with dotted names into
its params (diary.lr reaches the transformer's diary block). A setting must be one of the candidate's
params (its family's defaults, or set in its config), so a typo fails before anything trains:

    tune:
      seed: 0
      candidates:
        window_boosting:
          trials: 30
          space:
            learning_rate: {type: float, low: 0.01, high: 0.3, log: true}
            leaves:        {type: int, low: 7, high: 63}
            # also {type: categorical, choices: [...]}, and step for float and int

Each trial fits the candidate with the sampled settings merged over its config, on every fold of the
patient split's development patients, with tune.seed (experiments.cross_validate, as train does),
and returns its score on the selection metric (selection.metric). The same metric as selection, by
design: tuning for one score and choosing by another would optimise the wrong thing. A score is
maximised, or minimised if lower is better (an error, core/metrics.higher_is_better). The test
patients are never loaded. A transformer trial that only changes the diary step reuses each fold's
cached pretrained body (motor_causal_transformer/pretrained_cache.py), so it costs only that step.

The search is Optuna's TPE (Bayesian), seeded. Each candidate's study is kept in a SQLite file
(artifacts/tuning/studies.db), named by what it tunes on: the split, the data, the candidate's
config, the space, the metric and the seed. So a rerun resumes the same study (a crash loses only the
trial it was in, which the next run marks failed), and any change starts a new one instead of mixing
trials scored differently. A trial whose settings the model refuses (a ConfigError) is recorded as
failed and the search goes on; any other error stops the run.

MLflow, through the Tracker:

    tune                        tags: split hash, data fingerprint, metric
    └── <candidate>             params: the space; the best trial's settings (best.*) and cv scores,
        │                       run/tune_trials and run/tune_best_trial; artifact: the trials table
        └── trial-<n>           params: its settings; cv/<label>/<metric> (+ _fold_sd)
            └── seed-<s>-fold-<k>   as in train

Written locally:

    artifacts/tuning/studies.db         the studies (resumable; sensitivity_hparams.py reads them)
    artifacts/tuning/<candidate>.csv    every trial: number, state, score, settings, seconds
    artifacts/tuned_params.json         per candidate: the best settings and what they were tuned on;
                                        train uses them with train.use_tuned (experiments.read_tuned)

    python -m model_pipeline.steps.tune [--config config/model.yaml]
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import numbers
import sys
import time
import warnings
from pathlib import Path

from model_pipeline.core.errors import ConfigError, ModelError, TrackingError
from model_pipeline.core.metrics import higher_is_better
from model_pipeline.core.tracker import validate_metric_name
from model_pipeline.data.folds import read_split
from model_pipeline.experiments import (
    candidate_fingerprint, cross_validate, fresh, load_development, provenance, write_tuned)
from model_pipeline.models.registry import candidates
from model_pipeline.utils.logging_setup import get_logger

logger = get_logger(__name__)

TYPES = ("float", "int", "categorical")
STUDIES_FILE = "studies.db"


class TuneModels:
    """Step D: one Optuna study per candidate in tune.candidates. See the top of this file."""

    def __init__(self, cfg, tracker):
        self.cfg, self.tracker = cfg, tracker
        self.models = dict(candidates(cfg))
        self.settings = _settings(cfg, self.models)
        self.artifacts = Path(cfg["resolved_paths"]["artifacts_dir"])
        self.folder = self.artifacts / "tuning"

    def run(self, data=None, split=None):
        """Tune every candidate in tune.candidates.

        Args:
            data: A Dataset with the split's patients (tests pass one); None loads the development patients.
            split: The patient split; None reads paths.split_file.

        Returns:
            {candidate: {"params", "value", "trials", ...}}, as written to tuned_params.json.
        """
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)       # our own log says what each trial did
        split = split or read_split(self.cfg)
        names = list(self.settings["candidates"])
        if not names:
            logger.info("nothing to tune: tune.candidates is empty", extra={"event": "tune_nothing"})
            return {}
        if data is None:
            data, fingerprint = load_development(self.cfg, split, [(n, self.models[n]) for n in names])
        else:
            split.check(data.participants)
            fingerprint = "given"
        dev = data.subset(split.development)
        self.folder.mkdir(parents=True, exist_ok=True)
        metric = self.settings["metric"]
        out = {}
        code_tags, code_params = provenance()
        with self.tracker.run("tune", tags={"owner": "P2", "split_hash": split.hash, "data_fingerprint": fingerprint,
                                            "metric": metric, **code_tags}):
            trials = {n: c["trials"] for n, c in self.settings["candidates"].items()}
            self.tracker.log_params({"environment": code_params,
                                     "tune": {"seed": self.settings["seed"], "metric": metric, "candidates": trials}})
            for name in names:
                out[name] = self._candidate(name, dev, split, fingerprint)
                write_tuned(self.artifacts, {name: out[name]})     # after each, so a crash keeps the finished ones
        return out

    def _candidate(self, name, dev, split, fingerprint):
        import optuna
        template, config = self.models[name], self.settings["candidates"][name]
        metric, seed = self.settings["metric"], self.settings["seed"]
        key = _study_key(split.hash, fingerprint, candidate_fingerprint(template), config["space"], metric, seed)
        study_name = f"{name}-{key}"
        with warnings.catch_warnings():
            # The heartbeat marks a trial a crash left "running" as failed on resume. Optuna calls it
            # experimental; we rely on it on purpose.
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            storage = optuna.storages.RDBStorage(f"sqlite:///{(self.folder / STUDIES_FILE).as_posix()}",
                                                 heartbeat_interval=60, grace_period=180)
        direction = "maximize" if higher_is_better(metric) else "minimize"      # an error is minimised
        study = optuna.create_study(study_name=study_name, storage=storage, direction=direction, load_if_exists=True)
        # Seeded by how far the study got, so a resumed study doesn't redraw the settings it already tried.
        study.sampler = optuna.samplers.TPESampler(seed=seed + len(study.trials))
        with self.tracker.run(name, tags={"family": template.family, "study": study_name}):
            self.tracker.log_params({"space": config["space"], "trials": config["trials"]})
            with warnings.catch_warnings():                # experimental in Optuna, relied on on purpose
                warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
                optuna.storages.fail_stale_trials(study)   # a trial a crash left "running" is marked failed
            done = _complete(study)
            remaining = max(0, config["trials"] - len(done))
            logger.info("tuning %s: %d of %d trials done, %d to go", name, len(done), config["trials"], remaining,
                        extra={"event": "tune_started", "candidate": name, "study": study_name, "done": len(done),
                               "remaining": remaining})

            def objective(trial):
                return self._trial(trial, name, template, config["space"], dev, split)

            study.optimize(objective, n_trials=remaining, catch=(ConfigError,), gc_after_trial=True)
            done = _complete(study)
            if not done:
                raise ModelError(f"tuning {name}: no trial finished, so there are no settings to choose")
            best = study.best_trial
            self.tracker.log_params({"best": best.params})
            self.tracker.log_metrics({**best.user_attrs.get("metrics", {}), "run/tune_trials": len(done),
                                      "run/tune_best_trial": best.number})
            table = _trials_table(study, self.folder / f"{name}.csv")
            self.tracker.log_artifact(table, folder="tuning")
        logger.info("tuned %s: best %s %.4f (trial %d of %d)", name, metric, best.value, best.number, len(done),
                    extra={"event": "tune_finished", "candidate": name, "best_value": best.value,
                           "best_trial": best.number, "best_params": best.params})
        return {"params": nest(best.params), "flat_params": best.params, "metric": metric, "value": best.value,
                "trials": len(done), "best_trial": best.number, "study": study_name,
                "split_hash": split.hash, "data_hash": fingerprint, "candidate_hash": candidate_fingerprint(template),
                "tuned_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}

    def _trial(self, trial, name, template, space, dev, split):
        flat = {path: suggest(trial, path, spec) for path, spec in space.items()}
        model = fresh(template, nest(flat))                     # a ConfigError fails this trial only
        started = time.perf_counter()
        with self.tracker.run(f"trial-{trial.number}", tags={"trial": str(trial.number)}):
            self.tracker.log_params({"trial": flat})
            result = cross_validate(name, model, dev, split, [self.settings["seed"]], self.tracker)
            metrics = result.metrics()
            self.tracker.log_metrics(metrics)
        value = metrics.get(self.settings["metric"], math.nan)
        trial.set_user_attr("metrics", {k: v for k, v in metrics.items() if k.startswith("cv/")
                                        and not k.endswith("_fold_sd")})
        trial.set_user_attr("seconds", time.perf_counter() - started)
        logger.info("%s trial %d: %s %.4f in %.0f s", name, trial.number, self.settings["metric"], value,
                    time.perf_counter() - started, extra={"event": "tune_trial", "candidate": name,
                                                          "trial": trial.number, "value": value, "params": flat})
        return value                                            # NaN: Optuna records the trial as failed


# ---- search spaces ------------------------------------------------------------------------------

def suggest(trial, path, spec):
    """One setting's value for this trial, from its spec."""
    kind = spec["type"]
    if kind == "categorical":
        return trial.suggest_categorical(path, list(spec["choices"]))
    method = trial.suggest_float if kind == "float" else trial.suggest_int
    extra = {k: spec[k] for k in ("log", "step") if k in spec}
    return method(path, spec["low"], spec["high"], **extra)


def nest(flat):
    """{"diary.lr": 1} -> {"diary": {"lr": 1}}: dotted names into the nested settings."""
    out = {}
    for path, value in flat.items():
        node = out
        *parents, leaf = path.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return out


def _check_space(name, template, space):
    if not isinstance(space, dict) or not space:
        raise ConfigError(f"tune.candidates.{name}.space must name at least one setting")
    for path, spec in space.items():
        where = f"tune.candidates.{name}.space.{path}"
        node = template.params
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                raise ConfigError(f"{where}: {name} has no setting {path!r}")
            node = node[part]
        if isinstance(node, dict):
            raise ConfigError(f"{where}: {path!r} is a block of settings; name one setting inside it")
        if not isinstance(spec, dict) or spec.get("type") not in TYPES:
            raise ConfigError(f"{where}: type must be one of {TYPES}")
        if spec["type"] == "categorical":
            choices = spec.get("choices")
            if not isinstance(choices, list) or not choices or set(spec) - {"type", "choices"}:
                raise ConfigError(f"{where}: a categorical setting needs choices, a non-empty list, and nothing else")
            continue
        unknown = set(spec) - {"type", "low", "high", "log", "step"}
        low, high = spec.get("low"), spec.get("high")
        kind = numbers.Integral if spec["type"] == "int" else numbers.Real
        if unknown or not all(isinstance(v, kind) and not isinstance(v, bool) for v in (low, high)) or not low < high:
            raise ConfigError(f"{where}: needs low < high ({spec['type']}), and only log or step besides")
        if spec.get("log") and low <= 0:
            raise ConfigError(f"{where}: a log scale needs low above 0")
        if spec.get("log") and "step" in spec:
            raise ConfigError(f"{where}: Optuna can't take both log and step")


def _settings(cfg, models):
    section = cfg.get("tune")
    if not isinstance(section, dict):
        raise ConfigError("config needs a tune section: seed and candidates")
    metric = (cfg.get("selection") or {}).get("metric")
    try:
        validate_metric_name(metric)
    except TrackingError as e:
        raise ConfigError(f"selection.metric (what tuning maximises): {e}") from None
    if not metric.startswith("cv/"):
        raise ConfigError(f"selection.metric must be a cross-validation score (cv/...), not {metric!r}")
    if metric.endswith(("/hours", "_fold_sd")):
        raise ConfigError(f"selection.metric must be a score, not a count or a spread: {metric!r}")
    seed = section.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError(f"tune.seed must be a non-negative whole number, not {seed!r}")
    chosen = section.get("candidates") or {}
    if not isinstance(chosen, dict):
        raise ConfigError("tune.candidates must map candidate names to {trials, space}")
    unknown = sorted(set(chosen) - set(models))
    if unknown:
        raise ConfigError(f"tune.candidates names unknown candidate(s) {unknown}; known: {sorted(models)}")
    out = {}
    for name, config in chosen.items():
        if not isinstance(config, dict) or set(config) - {"trials", "space"}:
            raise ConfigError(f"tune.candidates.{name} takes trials and space only")
        trials = config.get("trials")
        if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
            raise ConfigError(f"tune.candidates.{name}.trials must be a whole number of 1 or more, not {trials!r}")
        _check_space(name, models[name], config.get("space"))
        out[name] = {"trials": trials, "space": config["space"]}
    return {"metric": metric, "seed": seed, "candidates": out}


# ---- helpers ------------------------------------------------------------------------------------

def _study_key(*parts):
    text = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.blake2b(text.encode("utf-8"), digest_size=6).hexdigest()


def _complete(study):
    import optuna
    return [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]


def _trials_table(study, path):
    """Every trial as a row: number, state, score, settings, seconds. Written through a temporary file."""
    import pandas as pd
    table = pd.DataFrame([{"trial": t.number, "state": t.state.name, "value": t.value,
                           "seconds": t.user_attrs.get("seconds"), **{f"param.{k}": v for k, v in t.params.items()}}
                          for t in study.trials])
    tmp = path.with_name(f".{path.name}.tmp")
    table.to_csv(tmp, index=False)
    tmp.replace(path)
    return path


def main(argv=None):
    from model_pipeline.tracking.mlflow_tracker import MlflowTracker
    from model_pipeline.utils.config import load_config
    from model_pipeline.utils.logging_setup import configure_logging

    parser = argparse.ArgumentParser(description="Tune candidates' settings (see model_pipeline/steps/tune.py)")
    parser.add_argument("--config", default=None, help="Path to a config YAML")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    configure_logging(cfg)
    for name, entry in TuneModels(cfg, MlflowTracker.from_config(cfg)).run().items():
        print(f"{name}: {entry['metric']} {entry['value']:.4f} after {entry['trials']} trials, {entry['flat_params']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
