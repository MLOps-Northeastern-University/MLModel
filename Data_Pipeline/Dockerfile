# The data pipeline's image: Python, git and the pinned libraries (DVC among them), and the code.
# Airflow isn't in it: Airflow (../airflow/) runs each step of the DAG in this image, with
# `python -m scripts.run_step <step>`, and the data folder mounted at /app/Data_Pipeline/data.
#
#   docker build -t data-pipeline --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
#   docker run --rm -v "$PWD/data:/app/Data_Pipeline/data" data-pipeline python -m scripts.pipeline
#
# docker-compose.yaml mounts the repo over /app instead, so code changes need no rebuild.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# git, for DVC: it records data versions next to the code.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && git config --global --add safe.directory '*'

WORKDIR /app/Data_Pipeline
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The commit the image was built from, for the logs: there's no git history in the image.
ARG GIT_COMMIT=""
ENV GIT_COMMIT=${GIT_COMMIT}

CMD ["python", "-m", "scripts.pipeline"]
