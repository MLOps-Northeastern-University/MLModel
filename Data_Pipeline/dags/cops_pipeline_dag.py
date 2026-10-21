"""Airflow DAG for the COPS Parkinson's data pipeline.

Flow:
    fetch_manifest -> download_demographics ----------------------------+
                   -> process_participant (one mapped task per person) -+-> merge
    merge -> clean -> validate -> detect_anomalies -> bias_report -> dvc_version -> notify

process_participant downloads, extracts features and deletes the zip in one
task. Splitting download and processing into two mapped tasks would make
Airflow download every zip before processing any, filling the disk in full
mode. max_active_tis_per_dag limits how many participants run at once.

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

from scripts import pipeline_steps as steps  # noqa: E402
from scripts.alerts import airflow_failure_callback, notify  # noqa: E402
from scripts.utils.config import load_config  # noqa: E402

PARALLEL_PARTICIPANTS = 4

default_args = {
    "owner": "mednova",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "on_failure_callback": airflow_failure_callback,
}


@dag(
    dag_id="cops_data_pipeline",
    description="COPS accelerometry + diary -> hourly features for kinesia prediction",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args=default_args,
    params={"mode": Param("sample", enum=["sample", "full"], description="sample = 5 smallest participants")},
    tags=["mlops", "parkinsons", "data-pipeline"],
)
def cops_data_pipeline():

    @task
    def fetch_manifest(params=None):
        cfg = load_config()
        return steps.step_manifest(cfg, params["mode"])

    @task
    def list_participants(manifest):
        return manifest["participants"]

    @task
    def download_demographics(manifest):
        return steps.step_demographics(load_config(), manifest)

    @task(max_active_tis_per_dag=PARALLEL_PARTICIPANTS, retries=2)
    def process_participant(entry):
        return steps.step_participant(load_config(), entry)

    @task
    def merge(results, demographics_path):
        return steps.step_merge(load_config(), list(results), demographics_path)

    @task
    def clean(merged_path):
        return steps.step_clean(load_config(), merged_path)

    @task
    def validate(features_path, run_id=None):
        # run_id is filled in by Airflow from the task context.
        steps.step_validate(load_config(), features_path, run_id or "airflow")
        return features_path

    @task(retries=0)
    def detect_anomalies(features_path):
        steps.step_anomalies(load_config(), features_path)
        return features_path

    @task
    def bias_report(features_path):
        return steps.step_bias(load_config(), features_path)

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
    def notify_success(report_path):
        cfg = load_config()
        notify("[COPS pipeline] Run succeeded", f"Bias report: {report_path}", cfg)

    manifest = fetch_manifest()
    demographics = download_demographics(manifest)
    results = process_participant.expand(entry=list_participants(manifest))
    features = validate(clean(merge(results, demographics)))
    report = bias_report(detect_anomalies(features))
    report >> dvc_version >> notify_success(report)


cops_data_pipeline()
