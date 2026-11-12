"""Tracking with MLflow.

Owner: P2.

Uses the MLflow server at MLFLOW_TRACKING_URI, or mlflow.tracking_uri in the
config. Credentials, if the server needs them, come only from MLflow's own
environment variables (MLFLOW_TRACKING_USERNAME / _PASSWORD / _TOKEN).

Creating the tracker checks the server once: if it can't be reached, it raises
TrackingError before any training starts. After that, MLflow's client retries
network errors itself, and a write that still fails fails the run.

    tracker = MlflowTracker.from_config(cfg)
    with tracker.run("train"):
        ...
"""

import os
import time

from mlflow import MlflowClient
from mlflow.entities import Metric, Param, RunTag
from mlflow.exceptions import MlflowException
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

from scripts.core.errors import TrackingError
from scripts.core.tracker import Tracker
from scripts.utils.logging_setup import get_logger

logger = get_logger(__name__)


class MlflowTracker(Tracker):
    """Records runs in one MLflow experiment."""

    def __init__(self, tracking_uri, experiment):
        super().__init__()
        self.client = MlflowClient(tracking_uri=tracking_uri)
        try:
            found = self.client.get_experiment_by_name(experiment)
            self.experiment_id = found.experiment_id if found else self.client.create_experiment(experiment)
        except (MlflowException, OSError) as exc:
            raise TrackingError(f"can't reach MLflow at {tracking_uri}: {exc}") from exc
        logger.info("tracking to MLflow", extra={
            "tracking_uri": tracking_uri, "experiment": experiment, "experiment_id": self.experiment_id})

    @classmethod
    def from_config(cls, cfg):
        """Build from the config's mlflow section; MLFLOW_TRACKING_URI overrides its tracking_uri."""
        settings = cfg["mlflow"]
        return cls(os.environ.get("MLFLOW_TRACKING_URI", settings["tracking_uri"]), settings["experiment"])

    def _start_run(self, name, parent_id, tags):
        if parent_id:
            tags = {**tags, MLFLOW_PARENT_RUN_ID: parent_id}   # how the MLflow UI shows nesting
        return self.client.create_run(self.experiment_id, run_name=name, tags=tags).info.run_id

    def _end_run(self, run_id, status):
        self.client.set_terminated(run_id, status.value)

    def _log_params(self, run_id, params):
        self.client.log_batch(run_id, params=[Param(k, str(v)) for k, v in params.items()])

    def _log_metrics(self, run_id, metrics, step):
        now_ms = int(time.time() * 1000)
        self.client.log_batch(run_id, metrics=[Metric(k, v, now_ms, step or 0) for k, v in metrics.items()])

    def _log_artifact(self, run_id, path, folder):
        if path.is_dir():
            self.client.log_artifacts(run_id, str(path), folder)
        else:
            self.client.log_artifact(run_id, str(path), folder)

    def _set_tags(self, run_id, tags):
        self.client.log_batch(run_id, tags=[RunTag(k, v) for k, v in tags.items()])
