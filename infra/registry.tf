# The team's Docker images: mlflow (the server, from Model_Pipeline/mlflow/), the airflow, data
# and model images model.yml builds for training runs, and the Backend's api image (backend_run.tf).
# There's no cleanup policy, so every pushed tag stays until someone deletes it.
resource "google_artifact_registry_repository" "ml" {
  repository_id = "ml"
  format        = "DOCKER"
  location      = var.region

  depends_on = [google_project_service.apis]
}

# Lets CI push the images it builds.
resource "google_artifact_registry_repository_iam_member" "training_ml" {
  repository = google_artifact_registry_repository.ml.name
  location   = var.region
  # Writer includes reading, so model.yml's images job can also check whether a tag is already
  # there (gcloud artifacts docker images describe) and skip that build.
  role   = "roles/artifactregistry.writer"
  member = "serviceAccount:${google_service_account.training.email}"
}

# Model bundles: plain files (model.tflite, metrics, config, model card), one version per
# model that passes the gates, as package parkinsons-risk, versions v1, v2, ...
resource "google_artifact_registry_repository" "bundles" {
  # Nothing in this repo pushes here yet: steps/register.py releases to the models bucket instead,
  # gs://team17-mlops-neu-models/parkinsons-risk/ (MODEL_REGISTRY, set on the training VM by
  # airflow/vm/startup.sh). This repository is the other choice model.yaml's paths.registry_dir names.
  repository_id = "bundles"
  format        = "GENERIC"
  location      = var.region

  depends_on = [google_project_service.apis]
}

# Lets CI push a bundle after a run passes the gates.
resource "google_artifact_registry_repository_iam_member" "training_bundles" {
  repository = google_artifact_registry_repository.bundles.name
  location   = var.region
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.training.email}"
}
