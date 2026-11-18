"""What the tests of a whole pipeline run share: the fake family, a ticking clock and a small config.

Used by test_run_model_pipeline.py (the whole run in one process) and test_run_step.py (each step
on its own, as the Airflow DAG runs them). Import the fixtures into a test module to use them.
"""

import datetime as dt
from types import SimpleNamespace

import pytest

from scripts.models import registry
from scripts.steps import train
from tests.helpers import make_config
from tests.steps.fake_models import pipeline_config

# The MLflow runs a whole run makes under model_pipeline, in order (register comes last, when it releases).
STEP_RUNS = ["train", "evaluate", "bias", "explain", "select_model", "compare_production", "export"]
# The release gate's rules that the fake family can be scored on (it predicts now only).
RULES = [{"metric": "cv/state_now/off_auc", "must_improve": True},
         {"metric": "cv/sleep_now/asleep_auc", "must_improve": False}]


@pytest.fixture(autouse=True)
def fake_family(monkeypatch):
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
    monkeypatch.setattr(train, "dt", SimpleNamespace(datetime=Ticking, timezone=dt.timezone))


def run_config(tmp_path):
    """The full config, with the fake family and small bias and explain settings.

    The fake data is biased on purpose; a gap limit of 1 lets a candidate through, so the run
    reaches the release (the tests that check the bias gate lower it). Its tremor is noise, so the
    validation gate checks only the OFF score.
    """
    small = pipeline_config(tmp_path / "artifacts", max_recall_gap=1.0)
    sections = {k: small[k] for k in ("models", "train", "bias", "explain")}
    config = make_config(tmp_path, evaluation={"bootstrap_resamples": 50}, release_gate={"rules": RULES},
                         **sections)
    config["validation_gate"]["min"] = {"cv/state_now/off_auc": 0.5}   # replaced, not merged into
    config["train"].update(use_tuned=False, charts={"enabled": False})  # nothing tuned here
    return config
