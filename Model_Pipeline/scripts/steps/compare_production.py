"""Compares the new model with the one in production: acts as a release gate

Owner: Ayush

Both models are scored on the same fixed test patients, on hours without a
check-in. The new model may replace the active one only if:

- it improves OFF-now AUC across the patient folds;
- it improves or holds OFF-now AUC for patients without DBS;
- it doesn't get worse for tremor, dyskinesia, asleep or next-hour OFF.

Otherwise it raises GateError, the run stops, and patients keep the current
model. The rules are mentioned in MODEL.md, "Release gate and retraining".
"""


from scripts.core.errors import GateError
from scripts.core.metrics import higher_is_better
from scripts.utils.logging_setup import get_logger


logger = get_logger(__name__)


def passes(new, production, must_improve, higher=True):
    """Whether the new model's score passes against production's score,
    given the must_improve flag.
    If must_improve is True, the new model must strictly improve over production.
    If must_improve is False, the new model can be equal to production.
    higher=False for a score where lower is better (an error): then improving means going down."""
    if not higher:
        new, production = -new, -production
    if must_improve:
        return new > production
    return new >= production


def failed_rules(rules, new_metrics, production_metrics):
    """The rules the new model fails, as readable messages.

    A metric missing from either model fails its rule: the gate can't pass
    without the score it checks.
    """
    failures = []
    for rule in rules:
        name = rule["metric"]
        if name not in new_metrics or name not in production_metrics:
            failures.append(f"{name}: missing")
            continue
        new, production = new_metrics[name], production_metrics[name]
        if not passes(new, production, rule["must_improve"], higher_is_better(name)):
            failures.append(f"{name}: {new:.3f} vs {production:.3f}")
    return failures


class CompareProduction:
    """The release gate: the new model against the active one."""

    def __init__(self, cfg, tracker):
        self.rules = cfg["release_gate"]["rules"]
        self.tracker = tracker

    def run(self, new_metrics, production_metrics):
        """Raise GateError unless the new model passes every rule."""
        with self.tracker.run("compare_production", tags={"owner": "P4"}):
            if production_metrics is None:
                self.tracker.set_tags({"release_gate": "first_model"})
                logger.info("No model in production yet, so the new one goes ahead")
                return
            failures = failed_rules(self.rules, new_metrics, production_metrics)
            self.tracker.set_tags({"release_gate": "failed" if failures else "passed"})
            if failures:
                raise GateError("production", "; ".join(failures))
            logger.info("The new model passes all %d rules", len(self.rules))
