"""The model pipeline as a DAG: the steps of Model_Pipeline/scripts/run_step.py, each in a container.

    start -> tune -> train -> evaluate -> bias -> explain -> select -> compare -> export -> register
        -> finish (every step passed) or finish_failed (a step failed)

start saves the run's choices and opens the model_pipeline run in MLflow; every step logs under it,
so MLflow shows one run with a child per step. tune does nothing unless the run tunes, and register
nothing unless it releases. A failed step (a gate, a crash) fails its task, the steps after it
don't run, and finish_failed closes the MLflow run as failed.

CI triggers it (a merge to main that bumps Model_Pipeline/config/retrain.yaml), and so can the
Backend on drift. Trigger config, all optional:

    {"candidates": ["motor_causal_transformer"], "tune": true, "reason": "deeper encoder", "release": true}
"""

from datetime import datetime, timedelta

from airflow.decorators import dag
from airflow.models.param import Param

from alerts import task_failed
from containers import model_container, step

RUN = "python -m scripts.run_step"
STEPS = ["tune", "train", "evaluate", "bias", "explain", "select", "compare", "export", "register"]

# start's options, from the trigger's config. Jinja fills them in when the task runs.
START = (
    f"{RUN} start"
    "{% if params.candidates %} --candidates {{ params.candidates | join(' ') }}{% endif %}"
    "{% if params.tune %} --tune{% endif %}"
    "{% if not params.release %} --no-register{% endif %}"
    "{% if params.reason %} --reason {{ params.reason | tojson }}{% endif %}"
)


@dag(
    dag_id="model_pipeline",
    description="Train, check and release the Parkinson's model: every gate, logged to MLflow",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    # A step is retried only by hand: a rerun repeats hours of training, and a gate won't pass twice.
    default_args={"owner": "mednova", "retries": 0, "on_failure_callback": task_failed},
    params={
        "candidates": Param([], type="array", description="Candidates to train (models.candidates); empty = all"),
        "tune": Param(False, type="boolean", description="Tune their settings again first (slow)"),
        "reason": Param("", type="string", description="Why this run: a change, drift"),
        "release": Param(True, type="boolean", description="Release the model if it passes every gate"),
    },
    max_active_runs=1,          # one run at a time: they share the artifacts folder
    tags=["mlops", "parkinsons", "model-pipeline"],
)
def model_pipeline():
    start = step("start", START, model_container())
    steps = [step(name, f"{RUN} {name}", model_container(), execution_timeout=timedelta(hours=8))
             for name in STEPS]
    finish = step("finish", f"{RUN} finish", model_container())
    finish_failed = step("finish_failed", f"{RUN} finish --status failed", model_container(),
                         trigger_rule="one_failed")

    previous = start
    for task in steps:
        previous >> task
        previous = task
    previous >> finish
    [start, *steps] >> finish_failed


model_pipeline()
