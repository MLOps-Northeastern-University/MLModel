# The hosted MLflow server's image (Cloud Run service mlflow, infra/mlflow.tf), apart from the
# pipeline's: it needs only the server, Postgres and GCS. Pushed to Artifact Registry ml as
# mlflow:<MLflow version>-<build number>.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# The same MLflow as the pipeline's client (requirements.txt) and the local server
# (docker-compose.yaml). psycopg2 is the Postgres driver the backend store URL names, and
# google-cloud-storage lets the server write files to the gs:// destination.
RUN pip install --no-cache-dir \
    mlflow==3.17.2 \
    psycopg2-binary==2.9.12 \
    google-cloud-storage==3.9.0

# The deploy sets these. DB_PASSWORD comes from Secret Manager (mlflow-db-password),
# CLOUD_SQL_CONNECTION is team17-mlops-neu:us-central1:mednova-db, and Cloud Run
# opens the database at /cloudsql/<connection>. Run history goes to Cloud SQL, files
# to the models bucket. Cloud Run picks the port. Two workers, so the UI's many
# requests per page aren't answered one at a time; the default four need more
# memory than the 1.5 GB the service gets.
ENV ARTIFACTS_DESTINATION=gs://team17-mlops-neu-models/mlflow \
    ALLOWED_HOSTS=*.run.app,localhost:*,127.0.0.1:*

# Shell form, so ${PORT} and the others are filled in, with exec, so mlflow replaces the shell
# and gets Cloud Run's stop signal itself. The empty host in the database URL and ?host= make
# psycopg2 connect through the Unix socket the Cloud SQL volume puts in /cloudsql/<connection>.
CMD exec mlflow server \
    --host=0.0.0.0 \
    --port="${PORT:-8080}" \
    --workers=2 \
    --backend-store-uri="postgresql+psycopg2://mlflow:${DB_PASSWORD}@/mlflow?host=/cloudsql/${CLOUD_SQL_CONNECTION}" \
    --artifacts-destination="${ARTIFACTS_DESTINATION}" \
    --allowed-hosts="${ALLOWED_HOSTS}"
