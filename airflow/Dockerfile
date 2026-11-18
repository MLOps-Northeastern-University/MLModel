# Airflow for both pipelines: Airflow, its Postgres driver and its Docker provider, and the DAGs.
# No pipeline code or libraries: each task runs in its pipeline's own image (dags/containers.py).
FROM apache/airflow:slim-2.10.5-python3.11

# Airflow's constraints file pins every library to versions tested with this Airflow, so the
# providers match it. Pinning apache-airflow too stops pip from changing Airflow itself.
ARG CONSTRAINTS=https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.11.txt
RUN pip install --no-cache-dir "apache-airflow[postgres]==2.10.5" apache-airflow-providers-docker \
        pytest --constraint "${CONSTRAINTS}"

COPY dags/ /opt/airflow/dags/
COPY tests/ /opt/airflow/tests/
