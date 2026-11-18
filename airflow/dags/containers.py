"""How both DAGs run a step: in its pipeline's Docker image, with the files mounted from this machine.

Airflow only orchestrates. Each task starts a container of the data or the model pipeline's image and
runs one command in it (that pipeline's scripts/run_step.py). Airflow never imports either pipeline,
so each keeps its own Python and libraries (the data pipeline 3.11, the model pipeline 3.13 with
PyTorch), and a step runs the same in a task as on a laptop.

The containers are started through this machine's Docker (its socket is mounted into Airflow), so
the folders they mount are paths on this machine, under its checkout of the repo:

    HOST_REPO/Data_Pipeline/data          the data: raw, interim, processed, reports
    HOST_REPO/Model_Pipeline/artifacts    the model pipeline's outputs and caches
    HARNET_DIR                            harnet10's weights (never in git or an image: its licence)

Settings, from the environment (airflow/.env, see .env.example):

    HOST_REPO      this machine's checkout of the MLModel repo
    DATA_IMAGE     the data pipeline's image                    (default data-pipeline:latest)
    MODEL_IMAGE    the model pipeline's image                   (default model-pipeline:latest)
    MODEL_GPU      1 to give the model's containers the GPU     (needs NVIDIA's container toolkit)
    HARNET_DIR     harnet10's folder on this machine
    and the alert, MLflow and registry settings in PASSED_ON, handed to every container.
"""

import os
import posixpath

from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import DeviceRequest, Mount

HOST_REPO = os.environ.get("HOST_REPO", "/srv/mednova/MLModel")
DATA_IMAGE = os.environ.get("DATA_IMAGE", "data-pipeline:latest")
MODEL_IMAGE = os.environ.get("MODEL_IMAGE", "model-pipeline:latest")
MODEL_GPU = os.environ.get("MODEL_GPU") == "1"
HARNET_DIR = os.environ.get("HARNET_DIR", "/srv/mednova/harnet10")

# Handed from Airflow's environment to every container, when set.
PASSED_ON = [
    "ALERT_EMAIL", "SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "SLACK_WEBHOOK_URL",   # alerts
    "MLFLOW_TRACKING_URI", "MODEL_MLFLOW_GOOGLE_AUTH",                                          # MLflow
    "MODEL_REGISTRY",                                                                           # the registry
    # Another config for this machine, as a path inside the container (a test with fewer participants).
    "COPS_CONFIG", "MODEL_CONFIG",
]

# Where each pipeline's files are inside its image (its Dockerfile's WORKDIR).
DATA_APP = "/app/Data_Pipeline"
MODEL_APP = "/app/Model_Pipeline"


def on_host(*parts):
    """A path under this machine's checkout of the repo."""
    return posixpath.join(HOST_REPO, *parts)


def passed_on():
    return {name: os.environ[name] for name in PASSED_ON if os.environ.get(name)}


def bind(source, target, read_only=False):
    return Mount(target=target, source=source, type="bind", read_only=read_only)


def container(image, mounts, environment=None, **extra):
    """The settings every task's container shares, for DockerOperator(...) or DockerOperator.partial(...)."""
    return dict(
        image=image,
        mounts=mounts,
        environment={**passed_on(), **(environment or {})},
        docker_url="unix://var/run/docker.sock",
        auto_remove="success",        # a failed one is kept, to look inside
        mount_tmp_dir=False,          # Airflow's temporary folder isn't on this machine's disk
        **extra,
    )


def data_container():
    """The data pipeline's image, with the data folder mounted."""
    return container(DATA_IMAGE, [bind(on_host("Data_Pipeline", "data"), f"{DATA_APP}/data")])


def repo_container():
    """The data pipeline's image with the whole checkout mounted: DVC needs git and its own files."""
    return container(DATA_IMAGE, [bind(HOST_REPO, "/app")], working_dir=DATA_APP)


def model_container():
    """The model pipeline's image, with the data (read-only), its outputs and harnet10, and the GPU if set."""
    mounts = [
        bind(on_host("Data_Pipeline", "data"), "/app/Data_Pipeline/data", read_only=True),
        bind(on_host("Model_Pipeline", "artifacts"), f"{MODEL_APP}/artifacts"),
        bind(HARNET_DIR, "/harnet10", read_only=True),
    ]
    environment = {"MODEL_HARNET_HUB": "/harnet10", "MODEL_LOG_FILE": "0"}   # logs go to Airflow, not a file
    gpu = [DeviceRequest(count=-1, capabilities=[["gpu"]])] if MODEL_GPU else None
    return container(MODEL_IMAGE, mounts, environment, device_requests=gpu)


def step(task_id, command, settings, **kwargs):
    """A task that runs command in a new container with these settings."""
    return DockerOperator(task_id=task_id, command=command, **settings, **kwargs)
