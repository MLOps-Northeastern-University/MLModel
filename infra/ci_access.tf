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

# ---- the Backend's deploys ----------------------------------------------------------------------

# The same pool trusts the Backend repository too, through a provider of its own.
resource "google_iam_workload_identity_pool_provider" "backend" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "backend"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
  }
  attribute_condition = "assertion.repository == '${var.backend_repo}'"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# What the Backend's deploy workflow logs in as. It pushes the api image and points the service
# and the scoring jobs at it; it never reads the database, the secrets or the buckets.
resource "google_service_account" "backend_deploy" {
  account_id   = "backend-deploy"
  display_name = "Backend deploys (CI)"

  depends_on = [google_project_service.apis]
}

# Updates the api service and the score-hours and check-drift jobs. On the whole project because
# they aren't made yet; a new revision also needs one of the two accounts below.
resource "google_project_iam_member" "backend_deploy_run" {
  project = var.project_id
  role    = "roles/run.developer"
  member  = "serviceAccount:${google_service_account.backend_deploy.email}"
}

resource "google_artifact_registry_repository_iam_member" "backend_deploy_ml" {
  repository = google_artifact_registry_repository.ml.name
  location   = var.region
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.backend_deploy.email}"
}

# A deploy sets the account the service or a job runs as, so it must be allowed to use that
# account: backend-api and backend-scoring (backend.tf), and no other.
resource "google_service_account_iam_member" "backend_deploy_acts_as" {
  for_each = {
    api     = google_service_account.backend_api.name
    scoring = google_service_account.backend_scoring.name
  }
  service_account_id = each.value
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.backend_deploy.email}"
}

# Lets workflows in the Backend repository act as backend-deploy.
resource "google_service_account_iam_member" "backend_deploy_wif" {
  service_account_id = google_service_account.backend_deploy.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.backend_repo}"
}
