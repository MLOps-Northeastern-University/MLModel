"""Airflow DAG for the COPS data pipeline: the same steps as `python -m scripts.pipeline`.

    fetch_manifest -> patients -> process_participant (one mapped task per participant)
        -> check_failures -> build -> run_checks -> dvc_version -> notify_success

patients comes first because each participant's wrist comes from the demographics.
process_participant downloads, processes and deletes one zip in one task, so the
disk never holds every zip at once; max_active_tis_per_dag limits how many run
at once. check_failures runs when every participant has finished or failed, and
stops the run only if more than data.max_failed_fraction failed.

Trigger with config {"mode": "sample"} or {"mode": "full"}.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.operators.bash import BashOperator

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from scripts import acquire, checks, pipeline  # noqa: E402
from scripts.utils import airflow_failure_callback, load_config, notify  # noqa: E402

# How many participants run at once: the same setting as the command line.
PARALLEL_PARTICIPANTS = load_config()["run"]["parallel_participants"]


@dag(
    dag_id="cops_data_pipeline",
    description="COPS diary + wrist accelerometry -> hours, doses, patients and 10-s windows for the model",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={"owner": "mednova", "retries": 1, "retry_delay": timedelta(minutes=2),
                  "on_failure_callback": airflow_failure_callback},
    params={"mode": Param("sample", enum=["sample", "full"],
                          description="sample = the smallest participants, full = all 66. "
                                      "Ignored while data.participants names a list in the config.")},
    tags=["mlops", "parkinsons", "data-pipeline"],
)
def cops_data_pipeline():

    @task
    def fetch_manifest(params=None):
        return acquire.fetch_manifest(load_config(), params["mode"])

    @task
    def patients(manifest):
        return pipeline.patients_step(load_config(), manifest)

    @task
    def list_participants(manifest):
        return manifest["participants"]

    @task(max_active_tis_per_dag=PARALLEL_PARTICIPANTS, retries=2)
    def process_participant(entry, patients_path):
        return pipeline.participant_step(load_config(), entry, patients_path)

    # Runs once every participant has finished or failed, so one participant that
    # still fails after its retries doesn't fail the whole run.
    @task(trigger_rule="all_done")
    def check_failures(manifest, results):
        return pipeline.check_failures(load_config(), manifest, list(results))

    @task
    def build(results, patients_path, params=None):
        return str(pipeline.build(load_config(), list(results), patients_path, params["mode"]))

    @task(retries=0)  # bad data won't get better by retrying
    def run_checks(processed):
        checks.run_checks(load_config())
        return processed

    dvc_version = BashOperator(
        task_id="dvc_version",
        bash_command=(
            f"cd {PIPELINE_ROOT} && "
            "if command -v dvc >/dev/null && dvc root >/dev/null 2>&1; then "
            "dvc commit -f build_dataset && (dvc push || echo 'dvc push skipped: no remote access'); "
            "else echo 'DVC not initialised; run dvc commit on the host'; fi"
        ),
    )

    @task
    def notify_success(processed):
        notify("[COPS pipeline] Run succeeded", f"Dataset: {processed}", load_config())

    manifest = fetch_manifest()
    patients_path = patients(manifest)
    results = process_participant.partial(patients_path=patients_path).expand(entry=list_participants(manifest))
    processed = run_checks(build(check_failures(manifest, results), patients_path))
    processed >> dvc_version >> notify_success(processed)


cops_data_pipeline()
