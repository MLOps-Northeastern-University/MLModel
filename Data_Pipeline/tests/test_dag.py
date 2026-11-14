"""The Airflow DAG: it loads, its tasks connect in the right order, and failures retry and alert.

Needs Airflow, which isn't in requirements.txt (the Airflow image brings it), so
these tests skip where Airflow isn't installed. Run them in the image:
    docker compose run --rm tests python -m pytest tests/test_dag.py
"""

from pathlib import Path

import pytest

pytest.importorskip("airflow")

from airflow.models import DagBag  # noqa: E402

from scripts.utils import airflow_failure_callback, load_config  # noqa: E402

DAGS = Path(__file__).resolve().parents[1] / "dags"


@pytest.fixture(scope="module")
def dag():
    bag = DagBag(dag_folder=str(DAGS), include_examples=False, read_dags_from_db=False)
    assert bag.import_errors == {}
    return bag.dags["cops_data_pipeline"]


def upstream(dag, task_id):
    return set(dag.get_task(task_id).upstream_task_ids)


def test_the_dag_has_every_step(dag):
    assert set(dag.task_ids) == {"fetch_manifest", "patients", "list_participants", "process_participant",
                                 "check_failures", "build", "run_checks", "dvc_version", "notify_success"}


def test_the_steps_run_in_order(dag):
    assert upstream(dag, "patients") == {"fetch_manifest"}
    assert upstream(dag, "list_participants") == {"fetch_manifest"}
    # Each participant's wrist comes from the patients table.
    assert upstream(dag, "process_participant") == {"list_participants", "patients"}
    assert upstream(dag, "check_failures") == {"fetch_manifest", "process_participant"}
    assert upstream(dag, "build") == {"check_failures", "patients"}
    assert upstream(dag, "run_checks") == {"build"}
    assert upstream(dag, "dvc_version") == {"run_checks"}
    # The data is versioned before success is announced.
    assert "dvc_version" in upstream(dag, "notify_success")


def test_failures_retry_and_alert(dag):
    for task in dag.tasks:
        assert task.on_failure_callback is airflow_failure_callback, task.task_id
    assert dag.get_task("process_participant").retries == 2  # downloads can fail for a moment
    assert dag.get_task("run_checks").retries == 0
    assert dag.get_task("build").retries == 1


def test_one_failed_participant_does_not_fail_the_run(dag):
    assert dag.get_task("check_failures").trigger_rule == "all_done"
    for task in ["build", "run_checks", "dvc_version"]:
        assert dag.get_task(task).trigger_rule == "all_success", task


def test_participants_run_in_parallel_as_configured(dag):
    expected = load_config()["run"]["parallel_participants"]
    assert dag.get_task("process_participant").max_active_tis_per_dag == expected


def test_the_dag_runs_only_when_triggered(dag):
    assert dag.schedule_interval is None
    assert dag.catchup is False
    assert dag.params["mode"] == "sample"
