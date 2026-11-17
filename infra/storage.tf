# Step 2: the buckets.

resource "google_storage_bucket" "buckets" {
  # training-data holds the DVC remote (cops-dvc/); models holds MLflow's files (mlflow/).
  for_each = toset(["training-data", "models"])
  name     = "${var.project_id}-${each.key}"
  location = var.region

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  depends_on = [google_project_service.apis]
}
