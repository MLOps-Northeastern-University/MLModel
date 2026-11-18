# The model pipeline's image: Python and the pinned libraries, nothing else.
# docker-compose mounts the repo over /app/Model_Pipeline, so code changes need
# no rebuild; the copied code is for running the image on its own (CI, the
# Airflow DAG on the GPU machine).
#
#   docker build -t model-pipeline .                                  # CPU: tests, laptops, CI
#   docker build -t model-pipeline:gpu --build-arg TORCH=cu128 \
#       --build-arg GIT_COMMIT=$(git rev-parse HEAD) .                # the GPU machine
#
# The GPU image runs with `docker run --gpus all`, on a machine with NVIDIA's
# driver and container toolkit. PyTorch's CUDA build brings CUDA itself.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# PyTorch's build: cpu, or a CUDA version (cu128, what the GPU run of 2026-11-15 used). CPU is the
# default, because the CUDA build is several GB that CI and laptops can't use.
ARG TORCH=cpu

WORKDIR /app/Model_Pipeline
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/${TORCH} \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# The commit the image was built from: there's no git in the image, so MLflow records this one.
ARG GIT_COMMIT=""
ENV GIT_COMMIT=${GIT_COMMIT}

CMD ["python", "-m", "model_pipeline.run_model_pipeline"]
