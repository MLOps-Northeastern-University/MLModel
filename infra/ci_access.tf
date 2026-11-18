resource "google_service_account" "training" {
  account_id   = "training"
  display_name = "ML training (CI)"

  depends_on = [google_project_service.apis]
}

# Read and write files in both buckets: dvc pull from training-data, MLflow's files in models.
resource "google_storage_bucket_iam_member" "training_buckets" {
  for_each = google_storage_bucket.buckets
  bucket   = each.value.name
  role     = "roles/storage.objectUser"
  member   = "serviceAccount:${google_service_account.training.email}"
}

# Workload Identity Federation: GitHub Actions logs in as training with a short-lived
# token instead of a key file.
resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github"
  display_name              = "GitHub Actions"

  depends_on = [google_project_service.apis]
}

# Trusts GitHub's ID tokens, but only from our repository.
resource "google_iam_workload_identity_pool_provider" "mlmodel" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "mlmodel"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
  }
  attribute_condition = "assertion.repository == '${var.github_repo}'"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# Lets workflows in that repository act as the training account.
resource "google_service_account_iam_member" "training_wif" {
  service_account_id = google_service_account.training.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repo}"
}
