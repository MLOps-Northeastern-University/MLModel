# Airflow's slim image: Airflow without most of its provider packages, which
# the pipeline doesn't use (and whose pins clash with its pandas version).
FROM apache/airflow:slim-2.10.5-python3.11

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /tmp/requirements.txt
# Pinning apache-airflow stops pip from up- or downgrading Airflow itself.
# The slim image has no Postgres driver, so the [postgres] extra adds it for
# Airflow's database.
RUN pip install --no-cache-dir "apache-airflow[postgres]==${AIRFLOW_VERSION}" -r /tmp/requirements.txt \
    && git config --global --add safe.directory '*'
