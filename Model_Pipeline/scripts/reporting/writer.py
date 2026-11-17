"""Saves the charts as PNGs and attaches them to the current tracker run.

Owner: P2.

    writer = ChartWriter.from_config(cfg, tracker)      # None when train.charts.enabled is false
    writer.predictions(frames, scores, "window_logistic/seed-0-fold-1", "window_logistic, fold 1")
    writer.comparison(results)                           # artifacts/charts/comparison/

Each call writes its PNGs to artifacts/charts/<where>/ (one folder per run, never nested in another
run's, since the whole folder is attached) and attaches that folder to the run that is
active, under charts/ in its artifacts (in MLflow: the run, then Artifacts, then charts). So every
run shows its own charts, and the local copies stay for the write-up.

A chart that fails to draw is logged as a warning (event charts_failed) and the run carries on: a
chart is a report on a model, never a reason to lose hours of training. The local folder is cleared
at the start of each train run (train.py), so it only ever holds one run's charts.
"""

import re
from pathlib import Path
from typing import Mapping, Optional, Sequence

from scripts.core.errors import ConfigError
from scripts.reporting import charts
from scripts.utils.logging_setup import get_logger

logger = get_logger(__name__)

FOLDER = "charts"
# The comparison chart's metrics when train.charts.compare isn't set: the selection metric's label,
# pooled and within, plus the other targets' headline scores.
DEFAULT_COMPARE = ("cv/state_now/off_auc", "cv/state_now/off_auc_within", "cv/state_now/score_spearman_within",
                   "cv/state_next_hour/off_auc_within", "cv/tremor_now/any_tremor_auc_within",
                   "cv/sleep_now/asleep_auc_within")


class ChartWriter:
    """Draws, saves and attaches charts (see the top of this file).

    Args:
        root: The local folder, artifacts/charts.
        tracker: The Tracker whose active run gets each chart.
        compare: The tracker metric names the comparison chart shows.
        folds: Whether each fold's run gets charts too (train.charts.folds).
    """

    def __init__(self, root, tracker, compare: Sequence[str] = DEFAULT_COMPARE, folds: bool = True):
        self.root, self.tracker = Path(root), tracker
        self.compare, self.folds = tuple(compare), folds

    @classmethod
    def from_config(cls, cfg, tracker) -> Optional["ChartWriter"]:
        """The writer train.charts asks for, or None if charts are off.

        Raises:
            ConfigError: train.charts isn't {enabled: bool, folds: bool, compare: [metric names]}.
        """
        section = (cfg.get("train") or {}).get("charts", {})
        if not isinstance(section, dict) or set(section) - {"enabled", "folds", "compare"}:
            raise ConfigError("train.charts takes enabled, folds and compare")
        for flag in ("enabled", "folds"):
            if not isinstance(section.get(flag, True), bool):
                raise ConfigError(f"train.charts.{flag} must be true or false")
        compare = section.get("compare", list(DEFAULT_COMPARE))
        if not isinstance(compare, list) or not compare or not all(isinstance(m, str) for m in compare):
            raise ConfigError("train.charts.compare must be a list of metric names, such as cv/state_now/off_auc")
        if not section.get("enabled", True):
            return None
        return cls(Path(cfg["resolved_paths"]["artifacts_dir"]) / FOLDER, tracker, compare, section.get("folds", True))

    def predictions(self, frames: Mapping, scores: Mapping, where: str, title: str) -> Optional[Path]:
        """A scorecard and one confusion matrix per label, for one run's predictions.

        Args:
            frames: {label: prediction rows}, from experiments.rows.
            scores: {label: {metric: value}} of those rows, from experiments.score.
            where: The folder under artifacts/charts, such as "<candidate>/seed-0-fold-1".
            title: What the charts are of, printed on them.

        Returns:
            The folder written, or None if nothing could be drawn.
        """
        drawings = {"scorecard": lambda: charts.scorecard_figure(scores, title)}
        for label, frame in frames.items():
            drawings[f"confusion_{label}"] = (
                lambda label=label, frame=frame: charts.confusion_figure(label, frame, title))
        return self._write(where, drawings)

    def comparison(self, results: Mapping[str, Mapping[str, float]], where: str = "comparison") -> Optional[Path]:
        """The candidates side by side on the compare metrics, with each one's spread across folds."""
        if not results:
            return None
        return self._write(where, {"comparison": lambda: charts.comparison_figure(
            results, self.compare, f"{len(results)} candidate(s), cross-validation")})

    def _write(self, where, drawings) -> Optional[Path]:
        if not _safe(where):
            raise ValueError("charts need a folder of their own under artifacts/charts")
        folder = self.root / _safe(where)
        folder.mkdir(parents=True, exist_ok=True)
        plt = charts.pyplot()
        written = []
        for name, draw in drawings.items():
            figure = None
            try:
                figure = draw()
                path = folder / f"{_safe(name)}.png"
                figure.savefig(path, dpi=charts.DPI, bbox_inches="tight")
                written.append(path)
            except Exception as e:      # a chart never stops a run: see the top of this file
                logger.warning("chart %s for %s not drawn: %s: %s", name, where or "the run", type(e).__name__, e,
                               extra={"event": "charts_failed", "chart": name, "where": where})
            finally:
                if figure is not None:
                    plt.close(figure)
        if not written:
            return None
        try:
            self.tracker.log_artifact(folder, folder=FOLDER)
        except Exception as e:
            logger.warning("charts for %s saved in %s but not attached to the run: %s: %s", where or "the run", folder,
                           type(e).__name__, e, extra={"event": "charts_failed", "where": where})
        return folder


def _safe(name: str) -> str:
    """A name safe in a path: letters, digits, - _ . and / between parts; never .. or absolute."""
    parts = [re.sub(r"[^A-Za-z0-9_.-]+", "_", p) for p in str(name).replace("\\", "/").split("/")]
    return "/".join(p for p in parts if p and p not in (".", ".."))


__all__ = ["DEFAULT_COMPARE", "FOLDER", "ChartWriter"]
