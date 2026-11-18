"""MLflowTracker against a real MLflow client and a temporary local store, no server."""

import pytest

pytest.importorskip("mlflow")

from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME  # noqa: E402

from scripts.core.errors import TrackingError  # noqa: E402
from scripts.core.tracker import RunStatus  # noqa: E402
from scripts.tracking.mlflow_tracker import MlflowTracker  # noqa: E402


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    return MlflowTracker(uri, "test-experiment")


def get_run(tracker, run_id):
    return tracker.client.get_run(run_id)


def test_records_params_metrics_tags_and_nesting(tracker):
    with tracker.run("train", tags={"owner": "P2"}) as parent:
        tracker.log_params({"model": {"family": "logistic", "C": 0.1}})
        with tracker.run("fold-1") as child:
            tracker.log_metric("val/state_now/off_auc", 0.6, step=0)
            tracker.log_metric("val/state_now/off_auc", 0.7, step=1)
            tracker.set_tags({"fold": 1})

    p, c = get_run(tracker, parent), get_run(tracker, child)
    assert p.data.params == {"model.family": "logistic", "model.C": "0.1"}
    assert p.data.tags["owner"] == "P2" and p.data.tags[MLFLOW_RUN_NAME] == "train"
    assert c.data.tags[MLFLOW_PARENT_RUN_ID] == parent and c.data.tags["fold"] == "1"
    history = tracker.client.get_metric_history(child, "val/state_now/off_auc")
    assert [(m.step, m.value) for m in history] == [(0, 0.6), (1, 0.7)]
    assert p.info.status == c.info.status == "FINISHED"


def test_failed_run_is_marked_failed(tracker):
    with pytest.raises(ValueError):
        with tracker.run("train") as run_id:
            raise ValueError("bad data")
    assert get_run(tracker, run_id).info.status == "FAILED"


def test_artifacts_file_and_folder(tracker, tmp_path):
    report = tmp_path / "report.csv"
    report.write_text("x\n")
    plots = tmp_path / "plots"
    plots.mkdir()
    (plots / "roc.txt").write_text("roc\n")
    with tracker.run("evaluate") as run_id:
        tracker.log_artifact(report)
        tracker.log_artifact(plots, folder="plots")
    names = {a.path for a in tracker.client.list_artifacts(run_id)}
    assert names == {"report.csv", "plots"}
    assert [a.path for a in tracker.client.list_artifacts(run_id, "plots")] == ["plots/roc.txt"]


def test_reuses_an_existing_experiment(tracker):
    again = MlflowTracker(tracker.client.tracking_uri, "test-experiment")
    assert again.experiment_id == tracker.experiment_id


def test_unreachable_server_fails_at_start(monkeypatch):
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")
    with pytest.raises(TrackingError, match="can't reach MLflow"):
        MlflowTracker("http://127.0.0.1:9", "test-experiment")


def test_from_config_prefers_the_environment(tmp_path, monkeypatch):
    uri = f"sqlite:///{(tmp_path / 'env.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    tracker = MlflowTracker.from_config({"mlflow": {"tracking_uri": "http://unused:1", "experiment": "e"}})
    assert tracker.client.tracking_uri == uri


def test_a_google_id_token_is_fetched_and_replaced_before_it_expires(tmp_path, monkeypatch):
    from google.oauth2 import id_token

    from scripts.tracking import mlflow_tracker
    fetched = []
    monkeypatch.setattr(id_token, "fetch_id_token", lambda request, audience: fetched.append(audience)
                        or f"token-{len(fetched)}")
    clock = [1000.0]
    monkeypatch.setattr(mlflow_tracker.time, "monotonic", lambda: clock[0])
    monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    tracker = MlflowTracker(uri, "e", google_auth=True)
    assert fetched == [uri] and mlflow_tracker.os.environ["MLFLOW_TRACKING_TOKEN"] == "token-1"
    with tracker.run("train"):                                 # within 45 minutes: the same token
        pass
    assert len(fetched) == 1
    clock[0] += mlflow_tracker.TOKEN_REFRESH_S + 1             # later: a new one before the next call
    with tracker.run("train"):
        pass
    assert len(fetched) == 2 and mlflow_tracker.os.environ["MLFLOW_TRACKING_TOKEN"] == "token-2"
    monkeypatch.delenv("MLFLOW_TRACKING_TOKEN")


def test_open_attach_and_close_a_run_across_processes(tracker):
    run_id = tracker.open_run("model_pipeline", tags={"owner": "P5"})
    other = MlflowTracker(tracker.client.tracking_uri, "test-experiment")   # as a new process would
    with other.attach(run_id), other.run("evaluate") as child:
        pass
    assert get_run(tracker, child).data.tags[MLFLOW_PARENT_RUN_ID] == run_id
    assert get_run(tracker, run_id).info.status == "RUNNING"            # attaching leaves it open
    other.close_run(run_id, RunStatus.FAILED)
    assert get_run(tracker, run_id).info.status == "FAILED"
