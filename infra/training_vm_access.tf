# What the training VMs may use in this project. The VMs run in their own project (compute/, for
# its GPU quota and credits), as the training-vm account made there; this grants that account the
# team's resources, nothing else.

# Read and write both buckets: in training-data, the DVC data, harnet10's weights (harnet10/), what
# runs keep (state/: the feature cache, tuned settings) and each run's result (runs/); in models,
# the registry.
resource "google_storage_bucket_iam_member" "training_vm_buckets" {
  for_each = google_storage_bucket.buckets
  bucket   = each.value.name
  role     = "roles/storage.objectUser"
  member   = "serviceAccount:${var.training_vm_email}"
}

# Log each run to the shared MLflow, which lets in only callers with run.invoker.
resource "google_cloud_run_v2_service_iam_member" "training_vm_mlflow" {
  name     = google_cloud_run_v2_service.mlflow.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${var.training_vm_email}"
}

# Pull the Airflow and pipeline images CI pushed.
resource "google_artifact_registry_repository_iam_member" "training_vm_ml" {
  repository = google_artifact_registry_repository.ml.name
  location   = var.region
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${var.training_vm_email}"
}
