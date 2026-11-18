"""The COPS data pipeline as a DAG: the steps of Data_Pipeline/scripts/run_step.py, each in a container.

    manifest -> patients -> participant (one task per participant, in parallel)
        -> check_failures -> build -> checks -> dvc_version -> notify_success

patients comes first because each participant's wrist comes from the demographics. Each
participant downloads, processes and deletes one zip, so the disk never holds every zip at once.
check_failures runs once every participant has finished or failed, and stops the run only if more
than data.max_failed_fraction failed (Data_Pipeline/config/pipeline.yaml).

Trigger with config {"mode": "sample"}, {"mode": "full"} or {"mode": "list"}.
"""

import json
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.providers.docker.operators.docker import DockerOperator

from alerts import task_failed
from containers import data_container, repo_container, step

# How many participants run at once. Each holds one participant's zip and windows in memory.
PARALLEL_PARTICIPANTS = 4

RUN = "python -m scripts.run_step"


@dag(
    dag_id="cops_data_pipeline",
    description="COPS diary + wrist accelerometry -> hours, doses, patients and 10-s windows for the model",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={"owner": "mednova", "retries": 1, "retry_delay": timedelta(minutes=2),
                  "on_failure_callback": task_failed},
    params={"mode": Param("sample", enum=["sample", "full", "list"],
                          description="sample = the smallest participants, full = all of them (the model's 60), "
                                      "list = the config's data.participants. The demo patients "
                                      "(data.exclude) are never processed.")},
    tags=["mlops", "parkinsons", "data-pipeline"],
)
def cops_data_pipeline():
    # Its last line of output, the participants' zip names as JSON, goes to XCom.
    manifest = step("manifest", f"{RUN} manifest --mode {{{{ params.mode }}}}", data_container(),
                    do_xcom_push=True)
    patients = step("patients", f"{RUN} patients", data_container())

    @task
    def participant_commands(names):
        return [f"{RUN} participant {name}" for name in json.loads(names)]

    participants = DockerOperator.partial(
        task_id="participant", max_active_tis_per_dag=PARALLEL_PARTICIPANTS,
        retries=2,                                  # downloads can fail for a moment
        **data_container(),
    ).expand(command=participant_commands(manifest.output))

    # Once every participant has finished or failed: one that still fails after its retries doesn't
    # fail the run, unless too many did.
    check_failures = step("check_failures", f"{RUN} check_failures", data_container(), trigger_rule="all_done")
    build = step("build", f"{RUN} build", data_container())
    checks = step("checks", f"{RUN} checks", data_container(), retries=0)   # bad data won't get better
    dvc_version = step("dvc_version", [
        "bash", "-c",
        "dvc commit -f build_dataset && (dvc push || echo 'dvc push skipped: no access to the remote')",
    ], repo_container())
    notify_success = step("notify_success", f"{RUN} notify_success", data_container())

    manifest >> patients >> participants
    participants >> check_failures >> build >> checks >> dvc_version >> notify_success


cops_data_pipeline()
