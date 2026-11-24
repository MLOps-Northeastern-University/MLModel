"""What the tests of a whole pipeline run share: the fake family, a ticking clock and a small config.

Used by test_run_model_pipeline.py (the whole run in one process) and test_run_step.py (each step
on its own, as the Airflow DAG runs them). Import the fixtures into a test module to use them.
"""

import datetime as dt
from types import SimpleNamespace

import pytest

from model_pipeline.models import registry
from model_pipeline.steps import train
from tests.helpers import make_config
from tests.steps.fake_models import pipeline_config

# The MLflow runs a whole run makes under model_pipeline, in order (register comes last, when it releases).
# These are the names the step classes give their tracker runs (select_model, compare_production), not
# the step names in pipeline_steps.STEPS. tune isn't listed: the one test that tunes doesn't check them.
STEP_RUNS = ["train", "evaluate", "bias", "explain", "select_model", "compare_production", "export"]
# The release gate's rules that the fake family can be scored on (it predicts now only).
# off_auc must strictly improve, which is what makes a second, identical run fail the release gate in
# these tests; asleep_auc only must not get worse.
RULES = [{"metric": "cv/state_now/off_auc", "must_improve": True},
         {"metric": "cv/sleep_now/asleep_auc", "must_improve": False}]


@pytest.fixture(autouse=True)
def fake_family(monkeypatch):
    """Register the fake family under the name its candidates use, so the steps build it like a real one."""
    monkeypatch.setitem(registry.FAMILIES, "fake_hourly", "tests.steps.fake_models:HourlyLogistic")


@pytest.fixture(autouse=True)
def ticking_clock(monkeypatch):
    """A clock a second further on at each call: train names each saved model by the second, and
    these tests run the pipeline twice within one."""
    class Ticking(dt.datetime):
        calls = 0

        @classmethod
        def now(cls, tz=None):
            cls.calls += 1
            return dt.datetime(2026, 1, 1, tzinfo=tz) + dt.timedelta(seconds=cls.calls)
    # train does `import datetime as dt` and calls dt.datetime.now(dt.timezone.utc), so swapping train.dt for
    # this namespace changes only train's clock, and timezone stays the real one.
    monkeypatch.setattr(train, "dt", SimpleNamespace(datetime=Ticking, timezone=dt.timezone))


def run_config(tmp_path):
    """The full config, with the fake family and small bias and explain settings.

    The fake data is biased on purpose; a gap limit of 1 lets a candidate through, so the run
    reaches the release (the tests that check the bias gate lower it). Its tremor is noise, so the
    validation gate checks only the OFF score.
    """
    small = pipeline_config(tmp_path / "artifacts", max_recall_gap=1.0)
    # Only these four sections come from the fake config; the rest stays as in model.yaml. That includes the
    # alerts, which are on there, so the test modules replace notify.
    sections = {k: small[k] for k in ("models", "train", "bias", "explain")}
    config = make_config(tmp_path, evaluation={"bootstrap_resamples": 50}, release_gate={"rules": RULES},
                         **sections)
    config["validation_gate"]["min"] = {"cv/state_now/off_auc": 0.5}   # replaced, not merged into
    config["train"].update(use_tuned=False, charts={"enabled": False})  # nothing tuned here
    return config
