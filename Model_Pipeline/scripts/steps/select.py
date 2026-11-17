"""Picks the model to release, after validation and bias.

Owner: Ayush

Keeps only the candidates that passed the validation and bias gates, then picks
the one with the best selection metric. The model runs on the server (MODEL.md,
"On the phone", so every saved model can be released.
When export.device.required is true, a candidate must also be exportable to the
phone. Raises GateError when no candidate is left.
"""

from scripts.core.errors import GateError
from scripts.core.metrics import higher_is_better
from scripts.utils.logging_setup import get_logger

logger = get_logger(__name__)


def eligible(candidates, need_device=False):
    """The candidates that passed the gates, and, with need_device, can run on the phone too."""
    return [c for c in candidates
            if c.get("gates_passed", False) and (not need_device or c.get("exportable", False))]


def best(candidates, metric):
    """The candidate with the best score on metric: the highest, or the lowest for an error
    (core/metrics.higher_is_better). If two candidates tie, the first one in the list is kept.
    Raises GateError if no candidate is eligible."""

    if not candidates:
        raise GateError("selection", "no eligible candidates found")

    pick = max if higher_is_better(metric) else min
    return pick(candidates, key=lambda c: c["metrics"][metric])


class SelectModel:
    """Picks the best eligible candidate and records the choice."""

    def __init__(self, cfg, tracker):
        self.metric = cfg["selection"]["metric"]
        self.need_device = cfg["export"]["device"]["required"]
        self.tracker = tracker

    def run(self, candidates):
        """The chosen candidate. Raises GateError when none is eligible."""
        with self.tracker.run("select_model", tags={"owner": "P4"}):
            kept = eligible(candidates, self.need_device)
            chosen = best(kept, self.metric)
            self.tracker.set_tags({"selected_model": chosen["name"]})
            logger.info("Selected %s: %s %.3f, from %d of %d candidates", chosen["name"],
                        self.metric, chosen["metrics"][self.metric], len(kept), len(candidates))
            return chosen
