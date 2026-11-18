"""Both DAGs: they load, their tasks connect in the right order, each runs in its pipeline's image,
and failures alert. Run in the Airflow image: docker compose run --rm tests"""

from pathlib import Path

import pytest
from airflow.models import DagBag

from alerts import task_failed
from containers import DATA_IMAGE, MODEL_IMAGE

DAGS = Path(__file__).resolve().parents[1] / "dags"


@pytest.fixture(scope="module")
def bag():
    loaded = DagBag(dag_folder=str(DAGS), include_examples=False, read_dags_from_db=False)
    assert loaded.import_errors == {}
    return loaded


def upstream(dag, task_id):
    return set(dag.get_task(task_id).upstream_task_ids)


def test_both_pipelines_are_dags_on_the_same_airflow(bag):
    assert set(bag.dag_ids) == {"cops_data_pipeline", "model_pipeline"}


# ---- the data pipeline -------------------------------------------------------------------

@pytest.fixture(scope="module")
def data(bag):
    return bag.dags["cops_data_pipeline"]


def test_the_data_steps_run_in_order(data):
    assert upstream(data, "patients") == {"manifest"}
    assert upstream(data, "participant_commands") == {"manifest"}
    assert upstream(data, "participant") == {"participant_commands", "patients"}
    assert upstream(data, "check_failures") == {"participant"}
    assert upstream(data, "build") == {"check_failures"}
    assert upstream(data, "checks") == {"build"}
    assert upstream(data, "dvc_version") == {"checks"}
    assert upstream(data, "notify_success") == {"dvc_version"}       # versioned before success is announced


def test_one_failed_participant_does_not_fail_the_run(data):
    assert data.get_task("check_failures").trigger_rule == "all_done"
    assert data.get_task("participant").partial_kwargs["retries"] == 2   # downloads can fail for a moment
    assert data.get_task("checks").retries == 0                          # bad data won't get better


def test_the_data_steps_run_in_the_data_pipelines_image(data):
    for task_id in ("manifest", "patients", "check_failures", "build", "checks", "dvc_version"):
        task = data.get_task(task_id)
        assert task.image == DATA_IMAGE, task_id
        assert task.on_failure_callback is task_failed, task_id
    assert data.get_task("participant").partial_kwargs["image"] == DATA_IMAGE


def test_the_data_dag_runs_only_when_triggered(data):
    assert data.schedule_interval is None and data.catchup is False
    assert data.params["mode"] == "sample"


# ---- the model pipeline ------------------------------------------------------------------

@pytest.fixture(scope="module")
def model(bag):
    return bag.dags["model_pipeline"]


STEPS = ["start", "tune", "train", "evaluate", "bias", "explain", "select", "compare", "export", "register",
         "finish"]


def test_the_model_steps_run_in_order(model):
    for before, after in zip(STEPS, STEPS[1:]):
        assert upstream(model, after) == {before}, after


def test_a_failed_step_closes_the_run_as_failed(model):
    failed = model.get_task("finish_failed")
    assert failed.trigger_rule == "one_failed"
    assert upstream(model, "finish_failed") == set(STEPS) - {"finish"}


def test_the_model_steps_run_in_the_model_pipelines_image_and_alert(model):
    for task in model.tasks:
        assert task.image == MODEL_IMAGE, task.task_id
        assert task.on_failure_callback is task_failed, task.task_id
        assert task.retries == 0, task.task_id                   # a rerun repeats hours of training


def test_the_run_options_become_starts_command(model):
    command = model.get_task("start").command
    for option in ("--candidates", "--tune", "--no-register", "--reason"):
        assert option in command
    assert model.max_active_runs == 1                            # runs share the artifacts folder
