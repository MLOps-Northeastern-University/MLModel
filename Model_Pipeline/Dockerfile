# The model pipeline's image: Python and the pinned libraries, nothing else.
# docker-compose mounts the repo over /app/Model_Pipeline, so code changes need
# no rebuild; the copied code is for running the image on its own (CI, the
# Airflow DAG on the GPU machine).
#
#   docker build -t model-pipeline .                                  # CPU: tests, laptops, CI
#   docker build -t model-pipeline:gpu --build-arg TORCH=cu129 \
#       --build-arg GIT_COMMIT=$(git rev-parse HEAD) .                # the GPU machine
#
# The GPU image runs with `docker run --gpus all`, on a machine with NVIDIA's
# driver and container toolkit. PyTorch's CUDA build brings CUDA itself.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# PyTorch's build: cpu, or a CUDA version (cu129: PyTorch's cu128 builds stop at 2.11, before the
# torch pin below; the GPU run of 2026-11-15 used cu128). CPU is the default, because the CUDA build
# is several GB that CI and laptops can't use.
ARG TORCH=cpu

# Airflow mounts the run's artifacts/ at /app/Model_Pipeline/artifacts (MODEL_APP in
# airflow/dags/containers.py), so this path and that one must stay the same.
WORKDIR /app/Model_Pipeline
# requirements.txt first and the code after: Docker reuses the slow install layer below until
# requirements.txt or TORCH changes, so a code change only redoes `COPY . .`.
COPY requirements.txt .
# torch first, from PyTorch's own index for the chosen build. requirements.txt names the same
# version, which pip then counts as installed, so the CPU or CUDA build stays. Keep the two pins
# equal: a different one there would make pip swap in PyPI's build instead.
RUN pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/${TORCH} \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# The commit the image was built from: there's no git in the image, so MLflow records this one.
ARG GIT_COMMIT=""
ENV GIT_COMMIT=${GIT_COMMIT}

# The whole pipeline by default. Airflow's tasks run one step each instead
# (python -m model_pipeline.run_step <step>), and the compose services set their own command.
CMD ["python", "-m", "model_pipeline.run_model_pipeline"]
