# Airflow

One Airflow for both pipelines, with two DAGs:

| DAG | What it runs | In which image |
|---|---|---|
| `cops_data_pipeline` | [`Data_Pipeline/scripts/run_step.py`](../Data_Pipeline/scripts/run_step.py): manifest → patients → participant (one task each, in parallel) → check_failures → build → checks → dvc_version → notify_success | the data pipeline's (Python 3.11) |
| `model_pipeline` | [`Model_Pipeline/scripts/run_step.py`](../Model_Pipeline/scripts/run_step.py): start → tune → train → evaluate → bias → explain → select → compare → export → register → finish | the model pipeline's (Python 3.13, PyTorch; the CUDA build on the GPU machine) |

Airflow only orchestrates. Each task starts a container of its pipeline's image and runs one step in it, so Airflow never imports either pipeline, each keeps its own Python and libraries, and a step runs the same in a task as on a laptop. The web UI shows each DAG's graph, each step's status, duration and live log, and the Gantt chart of a run. MLflow keeps the model's scores, charts and models.

```text
airflow/
├── Dockerfile            Airflow 2.10.5, its Postgres driver and Docker provider, and the DAGs
├── docker-compose.yaml   Postgres, the scheduler and the web UI; the DAG tests
├── .env.example          the settings: where the repo and harnet10 are, the images, MLflow, alerts
├── dags/
│   ├── containers.py     how a task runs its pipeline's image: the mounts, settings and GPU
│   ├── alerts.py         the email and Slack alert when a task fails
│   ├── data_pipeline.py
│   └── model_pipeline.py
└── tests/test_dags.py    both DAGs load, connect in order, and run in the right image
```

## How a task runs

Airflow starts each task's container through this machine's Docker (its socket is mounted into Airflow), so the folders a container mounts are this machine's, under its checkout of the repo (`HOST_REPO`):

| Folder on this machine | Mounted for | As |
|---|---|---|
| `Data_Pipeline/data/` | every data step; the model steps, read-only | `/app/Data_Pipeline/data` |
| the whole checkout | `dvc_version` (DVC needs git and its files) | `/app` |
| `Model_Pipeline/artifacts/` | every model step: outputs, the hand-off files, caches | `/app/Model_Pipeline/artifacts` |
| `HARNET_DIR` | every model step: harnet10's weights, never in git or an image (its licence) | `/harnet10` |

The steps of each pipeline hand over only through these files, never through Airflow, except the data DAG's participant list, which `manifest` prints for Airflow to start one task per participant.

## Run it

The pipelines' images first, from the repo's root (CI builds them and pushes them to Artifact Registry, tagged with the commit):

```bash
docker build -t data-pipeline Data_Pipeline
docker build -t model-pipeline Model_Pipeline                               # CPU
docker build -t model-pipeline --build-arg TORCH=cu128 Model_Pipeline       # the GPU machine
```

Then Airflow, from this folder:

```bash
cp .env.example .env                 # set HOST_REPO, HARNET_DIR and the rest
docker compose up airflow-init       # first time only: Airflow's database and login
docker compose up -d                 # the UI at http://localhost:8080
docker compose run --rm tests        # the DAG tests
```

On Docker Desktop for Windows, `HOST_REPO` is the repo's path as Docker sees it, such as `/run/desktop/mnt/host/d/path/to/MLModel`. For the model DAG on a laptop, start the model pipeline's MLflow first (`docker compose up -d mlflow` in `Model_Pipeline/`); the containers reach it at `http://host.docker.internal:5001`.

## Trigger a run

From the UI (Trigger DAG w/ config), or the REST API, which is how CI and the Backend start one:

```bash
curl -X POST "http://localhost:8080/api/v1/dags/model_pipeline/dagRuns" -u airflow:airflow \
  -H "Content-Type: application/json" \
  -d '{"conf": {"candidates": ["window_logistic"], "tune": false, "reason": "new data", "release": true}}'
```

| DAG | Config | Default |
|---|---|---|
| `cops_data_pipeline` | `mode`: `sample` or `full` | `sample` |
| `model_pipeline` | `candidates`: names from `models.candidates` | all |
| | `tune`: tune their settings again first (slow) | `false` |
| | `reason`: why this run, tagged in MLflow | none |
| | `release`: release the model if it passes every gate | `true` |

The model DAG runs one run at a time (the runs share `artifacts/`), and its steps aren't retried by themselves: a rerun repeats hours of training, and a failed gate won't pass a second time. A failed step fails its task, the steps after it don't run, and `finish_failed` closes the MLflow run as failed. Every failed task sends an alert, with a link to its log.
