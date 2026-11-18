"""Trains every candidate model on the same folds and seeds, and tracks all of it.

Owner: P2 (component C).

For each candidate in the config's models section (models/registry.py), on the patient split
(data/folds.py):

    1. cross-validation (experiments.cross_validate): for each seed and fold, fit on the fold's
       training patients, predict its validation patients, score them
    2. the final fit, on every development patient, with the epochs cross-validation chose (the
       median of the folds' best epochs, for a family with a final_epochs setting), saved under
       artifacts/models/
    3. the test patients, only when train.score_test is true: scored once, with the final model

With train.use_tuned, a candidate that tune.py tuned trains with the settings it chose (from
artifacts/tuned_params.json), merged over its config. Settings tuned on another split or other data
are reused with a warning (and tagged in MLflow) until a run tunes again; settings tuned on another
config of the candidate are skipped, so it trains with its config (experiments.read_tuned).

Before any candidate trains, the shuffled-label check (train.leakage_check): a model trained on
labels shuffled among the training hours must score near chance on the real labels. If it doesn't,
the labels are reaching the model through its inputs (for example through check-ins or event
fields built from the diary), and the run stops with a GateError. It can't catch a shortcut that
works with shuffled labels too, such as a patient ID or the clock: those need the ablations.

The test patients are scored once per patient split: a second score_test run on the same split is
refused (artifacts/test_scored/<split hash>.json records the first). Each run starts by removing the
last run's candidates.json and predictions, so a failed run never leaves stale ones behind.

MLflow, through the Tracker:

    train                       tags: split hash, data fingerprint, test scored; params: the config used;
    │                           chart: every candidate side by side (charts/comparison.png)
    ├── leakage_check-<name>    the shuffled-label scores
    └── <candidate>             tags: family, split hash, tuned; params: the settings it trained with;
        │                       cv/<label>/<metric> and its _fold_sd; for a model that pretrains, the
        │                       folds' mean validation skill per pretraining target and overall
        │                       (run/pretrain_val_skill) and its time, to compare one pretraining with
        │                       another (only between runs with the same split hash);
        │                       charts of every fold's validation hours together
        ├── seed-<s>-fold-<k>   val/<label>/<metric>, fit and predict seconds, the epoch curves; charts
        │                       of that fold (train.charts.folds)
        └── final               the saved model as an artifact; test/<label>/<metric> and charts if scored

Charts (reporting/, train.charts): each run above that has predictions gets a scorecard (every
score, pooled and within, against chance) and a confusion matrix per label, under charts/ in its
artifacts and in artifacts/charts/<candidate>/<run>/. A chart that fails to draw is a warning, never
a failed run.

Written locally for evaluate and bias, never sent to MLflow, since they're per patient-hour:

    artifacts/predictions/<label>.csv   participant_id, hour_index, fold (1..k, or test), seed, model,
                                        worn, y_true, then one column per class
    artifacts/candidates.json           per candidate: name, family, run id, model folder, metrics

    python -m model_pipeline.steps.train [--config config/model.yaml]
"""

import argparse
import datetime as dt
import math
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from model_pipeline.core.errors import ConfigError, GateError
from model_pipeline.core.targets import TARGETS, parse_label
from model_pipeline.data.folds import read_split
from model_pipeline.experiments import (
    concat, cross_validate, fresh, load_development, named, provenance, read_tuned, rows, score, write_predictions)
from model_pipeline.models.registry import candidates
from model_pipeline.reporting.writer import FOLDER as CHARTS, ChartWriter
from model_pipeline.utils.file_io import write_json
from model_pipeline.utils.logging_setup import get_logger, log_step

logger = get_logger(__name__)


class TrainModels:
    """Step C: cross-validate, refit and (once) test every candidate. See the top of this file."""

    def __init__(self, cfg, tracker):
        self.cfg, self.tracker = cfg, tracker
        self.settings = _settings(cfg)
        self.artifacts = Path(cfg["resolved_paths"]["artifacts_dir"])
        self.charts = ChartWriter.from_config(cfg, tracker)

    def run(self, data=None, split=None):
        """Train every candidate; return them as dicts for evaluate, bias and select.

        Args:
            data: A Dataset with every patient the split names (tests pass one); None loads it.
            split: The patient split; None reads paths.split_file.

        Returns:
            [{name, family, run_id, model_dir, metrics, split_hash, tuned, final_params}], in config order.

        Raises:
            GateError: The shuffled-label check scored above chance.
        """
        split = split or read_split(self.cfg)
        models = candidates(self.cfg)
        score_test = self.settings["score_test"]
        if data is None:
            data, fingerprint = load_development(self.cfg, split, models, include_test=score_test)
        else:
            split.check(data.participants)
            fingerprint = "given"
        tuned = read_tuned(self.artifacts, models, split, fingerprint) if self.settings["use_tuned"] else {}
        models = [(name, fresh(model, tuned[name]["params"]) if name in tuned else model) for name, model in models]
        marker = self.artifacts / "test_scored" / f"{split.hash}.json"
        if score_test and marker.exists():
            raise ConfigError(f"the test patients were already scored for split {split.hash} ({marker}). They're "
                              "scored once; delete that file only to score them again on purpose")
        self._clear_outputs()
        dev = data.subset(split.development)
        test = data.subset(split.test) if score_test else None
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
        code_tags, code_params = provenance()

        with self.tracker.run("train", tags={"owner": "P2", "split_hash": split.hash, "data_fingerprint": fingerprint,
                                             "test_scored": str(score_test).lower(), **code_tags}):
            self.tracker.log_params({"train": self.settings, "models": self.cfg["models"], "environment": code_params,
                                     "split": {"hash": split.hash, "test": len(split.test),
                                               "development": len(split.development), "folds": len(split.folds)},
                                     "tuned": {name: {k: entry.get(k) for k in ("study", "best_trial", "tuned_at",
                                                                                "stale")}
                                               for name, entry in tuned.items()}})
            if score_test:
                logger.warning("scoring the test patients: do this once, at the end of model work",
                               extra={"event": "test_scored"})
            by_name = dict(models)
            for name in self.settings["leakage_check"]["candidates"]:
                self._leakage_check(name, by_name[name], dev, split)
            frames, results, cv_metrics = {}, [], {}
            for name, model in models:
                result, model_frames, cv_metrics[name] = self._candidate(name, model, dev, test, split, stamp,
                                                                         name in tuned)
                results.append(result)
                for label, frame in model_frames.items():
                    frames.setdefault(label, []).append(frame)
            if self.charts is not None:
                self.charts.comparison(cv_metrics)
            write_predictions(self.artifacts / "predictions", frames)
            write_json(results, self.artifacts / "candidates.json")
            if score_test:
                write_json({"split_hash": split.hash, "scored_at": stamp, "candidates": [r["name"] for r in results]},
                           marker)
        return results

    def _clear_outputs(self):
        """Remove the last run's candidates.json, predictions and charts, so a run that fails halfway never
        leaves older outputs looking like its own (the saved models stay: each has its own dated folder)."""
        (self.artifacts / "candidates.json").unlink(missing_ok=True)
        if (self.artifacts / CHARTS).exists():           # even with charts off: old ones would look like this run's
            shutil.rmtree(self.artifacts / CHARTS)
        folder = self.artifacts / "predictions"
        if folder.exists():
            for path in folder.glob("*.csv"):
                path.unlink()

    def _candidate(self, name, template, dev, test, split, stamp, tuned):
        seeds = self.settings["seeds"]
        tags = {"family": template.family, "split_hash": split.hash, "tuned": str(tuned).lower()}
        with self.tracker.run(name, tags=tags) as run_id:
            self.tracker.log_params({"family": template.family, "targets": list(template.targets),
                                     "horizons": list(template.horizons), "params": template.params})
            result = cross_validate(name, template, dev, split, seeds, self.tracker, self.charts)
            self.tracker.log_metrics(result.metrics())
            if self.charts is not None:
                self.charts.predictions(result.frames, result.cv, f"{name}/cv",
                                        f"{name}, cross-validation (every fold's validation patients)")
            frames = [result.frames]

            final_params = {}
            if "final_epochs" in template.params and result.best_epochs:
                final_params["final_epochs"] = max(1, math.floor(float(np.median(result.best_epochs)) + 0.5))
            metrics = named("cv", result.cv)
            with self.tracker.run("final", tags={k: str(v) for k, v in final_params.items()}):
                model = fresh(template, final_params)
                model.fit(dev, seed=seeds[0])
                folder = model.save(self.artifacts / "models" / f"{name}-{stamp}")
                self.tracker.log_artifact(folder, folder=f"model/{name}")
                if test is not None:
                    test_frames = rows(name, model.predict(test), test, fold="test", seed=seeds[0])
                    test_scores = score(test_frames)
                    test_metrics = named("test", test_scores)
                    self.tracker.log_metrics(test_metrics)
                    if self.charts is not None:
                        self.charts.predictions(test_frames, test_scores, f"{name}/test",
                                                f"{name}, final model on the test patients")
                    metrics.update(test_metrics)
                    frames.append(test_frames)
        summary = ", ".join(f"{k} {v:.3f}" for k, v in metrics.items() if k.endswith("off_auc_within"))
        logger.info("trained %s: %s", name, summary,
                    extra={"event": "candidate_trained", "candidate": name, "final_params": final_params})
        return ({"name": name, "family": template.family, "run_id": run_id, "model_dir": str(folder),
                 "metrics": metrics, "split_hash": split.hash, "tuned": tuned, "final_params": final_params},
                concat(frames), result.metrics())

    def _leakage_check(self, name, template, dev, split):
        """Train on labels shuffled among the training hours; every AUC of the checked targets must stay
        within reach of chance.

        targets (train.leakage_check.targets) picks which targets are judged. Sleep can be left out: a model
        fitted to random labels still gives outputs that follow how much the wrist moves, and sleep hours
        barely move, so its sleep AUC lands far from 0.5 in a random direction with no leak at all. A leak
        through the inputs (events or check-ins built from the diary) would show in state and tremor.

        within_only (train.leakage_check.within_only) judges only the within-patient AUCs. The pooled
        ones mix patients, and patients who move less are OFF more often: a model fitted to random labels
        still follows how much each patient moves, so its pooled AUC lands far from 0.5 (0.36 on the real
        data) with no leak. Within one patient that effect is gone, and a leak would still raise the AUC.
        """
        limit = self.settings["leakage_check"]["max_auc"]
        targets = set(self.settings["leakage_check"]["targets"])
        judged = ("_auc_within",) if self.settings["leakage_check"]["within_only"] else ("_auc", "_auc_within")
        seed = self.settings["seeds"][0]
        rng = np.random.default_rng(seed)
        with self.tracker.run(f"leakage_check-{name}", tags={"family": template.family}), \
                log_step(logger, "leakage_check", candidate=name):
            parts = []
            for train_ids, val_ids in split.pairs():
                train = dev.subset(train_ids)
                shuffled = train.labels.iloc[rng.permutation(len(train.labels))].set_axis(train.labels.index)
                model = fresh(template)
                model.fit(replace(train, labels=shuffled), seed=seed)
                val = dev.subset(val_ids)
                parts.append(rows(name, model.predict(val), val, fold="leak", seed=seed))
            scores = score(concat(parts))
            self.tracker.log_metrics(named("cv", scores))
            off = [(f"{label}/{m}", v) for label, s in scores.items() if parse_label(label)[0] in targets
                   for m, v in s.items()
                   if m.endswith(judged) and math.isfinite(v) and abs(v - 0.5) > limit - 0.5]
            if off:
                raise GateError("leakage", f"{name} trained on shuffled labels scored "
                                + ", ".join(f"{m} {v:.3f}" for m, v in off) + f", beyond {1 - limit:.2f}-{limit:.2f}")
            logger.info("leakage check passed: %s on shuffled labels stays within %.2f-%.2f", name, 1 - limit, limit,
                        extra={"event": "leakage_check_passed", "candidate": name})


def _settings(cfg):
    section = cfg.get("train")
    if not isinstance(section, dict):
        raise ConfigError("config needs a train section: seeds, score_test, use_tuned, leakage_check")
    seeds = section.get("seeds")
    if not isinstance(seeds, list) or not seeds or not all(
            isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ConfigError(f"train.seeds must be a list of different non-negative whole numbers, not {seeds!r}")
    for flag in ("score_test", "use_tuned"):
        if not isinstance(section.get(flag, False), bool):
            raise ConfigError(f"train.{flag} must be true or false")
    if "score_test" not in section:
        raise ConfigError("train.score_test must be set, true or false")
    check = section.get("leakage_check") or {}
    names = check.get("candidates", [])
    limit = check.get("max_auc")
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ConfigError("train.leakage_check.candidates must be a list of candidate names")
    known = {c.get("name") for c in cfg.get("models", {}).get("candidates", []) if isinstance(c, dict)}
    unknown = sorted(set(names) - known)
    if unknown:
        raise ConfigError(f"train.leakage_check names unknown candidate(s) {unknown}")
    if names and (isinstance(limit, bool) or not isinstance(limit, (int, float)) or not 0.5 < limit < 1):
        raise ConfigError(f"train.leakage_check.max_auc must be between 0.5 and 1, not {limit!r}")
    targets = check.get("targets", list(TARGETS))
    if not isinstance(targets, list) or not targets or set(targets) - set(TARGETS):
        raise ConfigError(f"train.leakage_check.targets must be a list of targets from {sorted(TARGETS)}, "
                          f"not {targets!r}")
    within_only = check.get("within_only", False)
    if not isinstance(within_only, bool):
        raise ConfigError("train.leakage_check.within_only must be true or false")
    return {"seeds": seeds, "score_test": section["score_test"], "use_tuned": section.get("use_tuned", False),
            "leakage_check": {"candidates": names, "max_auc": float(limit) if names else None,
                              "targets": list(targets), "within_only": within_only}}


def main(argv=None):
    from model_pipeline.tracking.mlflow_tracker import MlflowTracker
    from model_pipeline.utils.config import load_config
    from model_pipeline.utils.logging_setup import configure_logging

    parser = argparse.ArgumentParser(description="Train every candidate model (see model_pipeline/steps/train.py)")
    parser.add_argument("--config", default=None, help="Path to a config YAML")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    configure_logging(cfg)
    results = TrainModels(cfg, MlflowTracker.from_config(cfg)).run()
    for r in results:
        within = sorted((k, v) for k, v in r["metrics"].items() if k.endswith("off_auc_within"))
        print(f"{r['name']}: " + ", ".join(f"{k} {v:.3f}" for k, v in within))
    return 0


if __name__ == "__main__":
    sys.exit(main())
