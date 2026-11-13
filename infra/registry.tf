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
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.training.email}"
}

# Model bundles: plain files (model.tflite, metrics, config, model card), one version per
# model that passes the gates, as package parkinsons-risk, versions v1, v2, ...
resource "google_artifact_registry_repository" "bundles" {
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
