"""How much each tuned setting affects performance, read from the tuning studies.

Owner: P2 (component G, hyperparameters).

For each candidate tune.py tuned (artifacts/tuned_params.json names its study), from that study's
finished trials:

    importance   each setting's share of how much the score varied across trials (fANOVA, seeded):
                 near 0 means the score hardly cares, so its value matters little; shares add up to 1.
                 If every trial scored the same, no setting mattered: every importance is 0 (and
                 importance.json says flat), with no importance plot
    plots        importance.png   the importances as bars
                 slice.png        the score of every trial against each setting
                 history.png      the score trial by trial, with the best so far

Nothing is trained: it reads only what tuning already recorded, so it takes seconds. The plots are
matplotlib, which needs no display and no browser. For a live look at the same studies, optuna-dashboard
reads artifacts/tuning/studies.db.

MLflow, through the Tracker:

    sensitivity_hparams
    └── <candidate>     run/importance_<setting> for each setting; artifacts: the plots and importance.json

Written locally: artifacts/sensitivity/<candidate>/ with the plots and importance.json.

    python -m scripts.steps.sensitivity_hparams [--config config/model.yaml]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

from scripts.core.errors import ConfigError
from scripts.experiments import TUNED_FILE, slug
from scripts.steps.tune import STUDIES_FILE
from scripts.utils.file_io import write_json
from scripts.utils.logging_setup import get_logger

logger = get_logger(__name__)

MIN_TRIALS = 3          # fewer finished trials say nothing about which setting matters


class HyperparameterSensitivity:
    """Step G (hyperparameters): importance and plots per tuned candidate. See the top of this file."""

    def __init__(self, cfg, tracker):
        self.cfg, self.tracker = cfg, tracker
        self.artifacts = Path(cfg["resolved_paths"]["artifacts_dir"])
        self.seed = (cfg.get("tune") or {}).get("seed", 0)

    def run(self):
        """{candidate: {"importance": {setting: share}, "trials": n, "folder": path}} for each tuned candidate.

        Raises:
            ConfigError: There's no tuned_params.json (run tune first).
        """
        tuned_file = self.artifacts / TUNED_FILE
        if not tuned_file.exists():
            raise ConfigError(f"no {tuned_file}: run tune before the sensitivity step")
        tuned = json.loads(tuned_file.read_text(encoding="utf-8"))
        storage = f"sqlite:///{(self.artifacts / 'tuning' / STUDIES_FILE).as_posix()}"
        out = {}
        with self.tracker.run("sensitivity_hparams", tags={"owner": "P2"}):
            for name, entry in tuned.items():
                result = self._candidate(name, entry, storage)
                if result is not None:
                    out[name] = result
        return out

    def _candidate(self, name, entry, storage):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.load_study(study_name=entry["study"], storage=storage)
        done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(done) < MIN_TRIALS:
            logger.warning("%s: only %d finished trial(s), too few to say which setting matters; skipped", name,
                           len(done), extra={"event": "sensitivity_skipped", "candidate": name, "trials": len(done)})
            return None
        folder = self.artifacts / "sensitivity" / name
        folder.mkdir(parents=True, exist_ok=True)
        evaluator = optuna.importance.FanovaImportanceEvaluator(seed=self.seed)
        values = [t.value for t in done]
        flat = max(values) - min(values) <= 1e-12          # fANOVA can't rank settings that changed nothing
        if flat:
            importance = {k: 0.0 for k in study.best_trial.params}
            logger.warning("%s: every trial scored %.4f, so no tuned setting changed the score", name, values[0],
                           extra={"event": "sensitivity_flat", "candidate": name})
        else:
            importance = {k: float(v) for k, v in optuna.importance.get_param_importances(
                study, evaluator=evaluator).items()}
        write_json({"candidate": name, "study": entry["study"], "metric": entry.get("metric"), "trials": len(done),
                    "flat": flat, "importance": importance, "best": entry.get("flat_params")},
                   folder / "importance.json")
        _plots(study, None if flat else evaluator, folder)
        with self.tracker.run(name, tags={"study": entry["study"]}):
            self.tracker.log_metrics({f"run/importance_{slug(k)}": v for k, v in importance.items()})
            self.tracker.log_artifact(folder, folder="sensitivity")
        ranked = ", ".join(f"{k} {v:.2f}" for k, v in importance.items())
        logger.info("%s: setting importance %s (from %d trials)", name, ranked, len(done),
                    extra={"event": "sensitivity_done", "candidate": name, "importance": importance})
        return {"importance": importance, "trials": len(done), "folder": str(folder)}


def _plots(study, evaluator, folder):
    """importance.png (unless evaluator is None), slice.png and history.png in folder, drawn without a display."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from optuna.visualization import matplotlib as plots

    drawings = {"slice": lambda: plots.plot_slice(study), "history": lambda: plots.plot_optimization_history(study)}
    if evaluator is not None:
        drawings["importance"] = lambda: plots.plot_param_importances(study, evaluator=evaluator)
    for name, draw in drawings.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")              # Optuna marks its matplotlib plots experimental
            axes = draw()
        figure = (axes.flat[0] if hasattr(axes, "flat") else axes).figure
        figure.savefig(folder / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(figure)


def main(argv=None):
    from scripts.tracking.mlflow_tracker import MlflowTracker
    from scripts.utils.config import load_config
    from scripts.utils.logging_setup import configure_logging

    parser = argparse.ArgumentParser(description="Hyperparameter sensitivity from the tuning studies")
    parser.add_argument("--config", default=None, help="Path to a config YAML")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    configure_logging(cfg)
    for name, result in HyperparameterSensitivity(cfg, MlflowTracker.from_config(cfg)).run().items():
        print(f"{name} ({result['trials']} trials): "
              + ", ".join(f"{k} {v:.2f}" for k, v in result["importance"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
