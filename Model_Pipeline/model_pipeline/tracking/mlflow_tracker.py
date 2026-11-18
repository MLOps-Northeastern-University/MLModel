"""Tracking with MLflow.

Owner: P2.

Uses the MLflow server at MLFLOW_TRACKING_URI, or mlflow.tracking_uri in the
config. Credentials, if the server needs them, come from MLflow's own
environment variables (MLFLOW_TRACKING_USERNAME / _PASSWORD / _TOKEN).

The team's MLflow runs on Cloud Run, which lets in only callers with a Google ID
token, and a token lasts an hour. CI passes one in MLFLOW_TRACKING_TOKEN, enough
for its run. A longer run (training on the GPU machine takes hours) sets
MODEL_MLFLOW_GOOGLE_AUTH=1 instead: the tracker then gets a token from the
machine's own Google identity (its service account) and replaces it every 45
minutes. MLflow reads MLFLOW_TRACKING_TOKEN on every request, so a new one takes
effect at once.

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

from model_pipeline.core.errors import TrackingError
from model_pipeline.core.tracker import Tracker
from model_pipeline.utils.logging_setup import get_logger

logger = get_logger(__name__)

# A Google ID token lasts an hour: a new one is fetched well before it runs out.
TOKEN_REFRESH_S = 45 * 60


class GoogleIdToken:
    """Keeps MLFLOW_TRACKING_TOKEN a fresh Google ID token for the MLflow server at audience."""

    def __init__(self, audience):
        self.audience = audience
        self.fetched_at = None

    def refresh_if_old(self):
        if self.fetched_at is not None and time.monotonic() - self.fetched_at < TOKEN_REFRESH_S:
            return
        from google.auth.transport.requests import Request
        from google.oauth2 import id_token
        os.environ["MLFLOW_TRACKING_TOKEN"] = id_token.fetch_id_token(Request(), self.audience)
        self.fetched_at = time.monotonic()
        logger.info("got a new Google ID token for MLflow", extra={"event": "mlflow_token_refreshed"})


class MlflowTracker(Tracker):
    """Records runs in one MLflow experiment."""

    def __init__(self, tracking_uri, experiment, google_auth=False):
        super().__init__()
        # The ID token's audience is the server's own address, as Cloud Run expects.
        self.token = GoogleIdToken(tracking_uri) if google_auth else None
        self._keep_token_fresh()
        # Ending a run makes MLflow fetch it again and print its link with an emoji. On a Windows
        # console that isn't UTF-8 the emoji crashes the run's end, and every fold would pay the extra
        # request; our own log already says which run ended.
        os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")
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
        return cls(os.environ.get("MLFLOW_TRACKING_URI", settings["tracking_uri"]), settings["experiment"],
                   google_auth=os.environ.get("MODEL_MLFLOW_GOOGLE_AUTH") == "1")

    def _keep_token_fresh(self):
        """Before each call to the server: a new ID token if the last is getting old."""
        if self.token is not None:
            self.token.refresh_if_old()

    def _start_run(self, name, parent_id, tags):
        self._keep_token_fresh()
        if parent_id:
            tags = {**tags, MLFLOW_PARENT_RUN_ID: parent_id}   # how the MLflow UI shows nesting
        return self.client.create_run(self.experiment_id, run_name=name, tags=tags).info.run_id

    def _end_run(self, run_id, status):
        self._keep_token_fresh()
        self.client.set_terminated(run_id, status.value)

    def _log_params(self, run_id, params):
        self._keep_token_fresh()
        self.client.log_batch(run_id, params=[Param(k, str(v)) for k, v in params.items()])

    def _log_metrics(self, run_id, metrics, step):
        self._keep_token_fresh()
        now_ms = int(time.time() * 1000)
        self.client.log_batch(run_id, metrics=[Metric(k, v, now_ms, step or 0) for k, v in metrics.items()])

    def _log_artifact(self, run_id, path, folder):
        self._keep_token_fresh()
        if path.is_dir():
            self.client.log_artifacts(run_id, str(path), folder)
        else:
            self.client.log_artifact(run_id, str(path), folder)

    def _set_tags(self, run_id, tags):
        self._keep_token_fresh()
        self.client.log_batch(run_id, tags=[RunTag(k, v) for k, v in tags.items()])
