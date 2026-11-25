# The Backend's own database on mednova-db (database.tf), next to MLflow's, and the two settings it
# can't start without. Both are kept in Secret Manager, so they're never in code, config or GitHub;
# the Backend's Cloud Run service and scoring jobs read them as DATABASE_URL and SECRET_KEY.
# Then the two accounts they run as, and what each may use.

resource "google_sql_database" "mednova" {
  name     = "mednova"
  instance = google_sql_database_instance.main.name

  # It holds the patients' data: a `terraform destroy` leaves it in place instead of deleting it.
  deletion_policy = "ABANDON"
}

# Letters and digits only, so the password can go straight into the database URL.
resource "random_password" "mednova_db" {
  length  = 32
  special = false
}

# The Backend's own login, apart from MLflow's: each has its own password and secret, so one
# can be changed without touching the other.
resource "google_sql_user" "mednova" {
  name     = "mednova"
  instance = google_sql_database_instance.main.name
  password = random_password.mednova_db.result
}

# The URL app/db.py hands to SQLAlchemy. On Cloud Run the database is reached through the folder
# the service's cloudsql volume mounts (/cloudsql/<connection name>), not over the network.
resource "google_secret_manager_secret" "backend_database_url" {
  secret_id = "backend-database-url"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

# postgresql+psycopg is SQLAlchemy's name for the psycopg 3 driver the Backend uses. The
# password sits inside the URL, so this one secret is all the Backend needs to connect.
resource "google_secret_manager_secret_version" "backend_database_url" {
  secret      = google_secret_manager_secret.backend_database_url.id
  secret_data = "postgresql+psycopg://${google_sql_user.mednova.name}:${random_password.mednova_db.result}@/${google_sql_database.mednova.name}?host=/cloudsql/${google_sql_database_instance.main.connection_name}"
}

# The key app/security.py scrambles enrollment codes with before saving them. Never replace it:
# every code already handed out would stop matching.
resource "random_password" "backend_secret_key" {
  length  = 48
  special = false
}

resource "google_secret_manager_secret" "backend_secret_key" {
  secret_id = "backend-secret-key"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "backend_secret_key" {
  secret      = google_secret_manager_secret.backend_secret_key.id
  secret_data = random_password.backend_secret_key.result
}

# The Firebase web app's API key (FIREBASE_WEB_API_KEY), which emails new staff the link to choose
# a password. Not secret in itself (it ships in Careportal's page), but kept out of git and CI all
# the same. Terraform makes the empty secret; its value is added by hand, once (variables.tf,
# firebase_web_api_key_added), so the key is never in code or the state.
resource "google_secret_manager_secret" "backend_firebase_web_api_key" {
  secret_id = "backend-firebase-web-api-key"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

# ---- who the Backend runs as --------------------------------------------------------------------

# Two accounts, so the public API never holds what the jobs need: the API service runs as
# backend-api, the scoring jobs (score-hours, check-drift) as backend-scoring.
resource "google_service_account" "backend_api" {
  account_id   = "backend-api"
  display_name = "Backend API (Cloud Run)"

  depends_on = [google_project_service.apis]
}

resource "google_service_account" "backend_scoring" {
  account_id   = "backend-scoring"
  display_name = "Backend scoring jobs (Cloud Run)"

  depends_on = [google_project_service.apis]
}

locals {
  backend_accounts = {
    api     = google_service_account.backend_api.email
    scoring = google_service_account.backend_scoring.email
  }
  backend_secrets = {
    database_url = google_secret_manager_secret.backend_database_url.id
    secret_key   = google_secret_manager_secret.backend_secret_key.id
  }
}

# Both connect to mednova-db. Cloud SQL can only grant this on the whole project; the password in
# backend-database-url is what keeps them to the mednova database.
resource "google_project_iam_member" "backend_sql" {
  for_each = local.backend_accounts
  project  = var.project_id
  role     = "roles/cloudsql.client"
  member   = "serviceAccount:${each.value}"
}

# Both read the two settings, and no other secret.
resource "google_secret_manager_secret_iam_member" "backend_settings" {
  for_each = {
    for pair in setproduct(keys(local.backend_accounts), keys(local.backend_secrets)) :
    "${pair[0]}/${pair[1]}" => { account = local.backend_accounts[pair[0]], secret = local.backend_secrets[pair[1]] }
  }
  secret_id = each.value.secret
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${each.value.account}"
}

# Only the API sends staff their password email, so only backend-api reads the web key.
resource "google_secret_manager_secret_iam_member" "api_firebase_web_api_key" {
  secret_id = google_secret_manager_secret.backend_firebase_web_api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.backend_api.email}"
}

# ---- what only the scoring jobs read and write --------------------------------------------------

# The registry: active.json and the released bundles (MODEL_REGISTRY).
resource "google_storage_bucket_iam_member" "scoring_models" {
  bucket = google_storage_bucket.buckets["models"].name
  # Viewer: the jobs only read the active bundle. Only a training run or a rollback by hand
  # (steps/register.py) writes to the registry, and backend-api gets no access to it at all.
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.backend_scoring.email}"
}

locals {
  # The prefix every object's name has in an IAM condition. Both conditions below start from it.
  training_data_objects = "projects/_/buckets/${google_storage_bucket.buckets["training-data"].name}/objects"
}

# harnet10's weights (HARNET_DIR), and nothing else in the bucket: the licensed data lives there
# too. A list is checked on the bucket, not an object, so the second line lets the jobs list this
# folder only; copying the weights lists it first.
resource "google_storage_bucket_iam_member" "scoring_harnet10" {
  bucket = google_storage_bucket.buckets["training-data"].name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.backend_scoring.email}"

  condition {
    title = "harnet10-only"
    expression = trimspace(<<-EOT
      resource.name.startsWith("${local.training_data_objects}/harnet10/")
      || api.getAttribute("storage.googleapis.com/objectListPrefix", "").startsWith("harnet10/")
    EOT
    )
  }
}

# The nightly drift check's reports (DRIFT_REPORTS), which MLModel's drift.yml turns into issues.
# User, not Creator: checking a night again overwrites that night's report.
resource "google_storage_bucket_iam_member" "scoring_drift" {
  bucket = google_storage_bucket.buckets["training-data"].name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.backend_scoring.email}"

  condition {
    title      = "drift-reports-only"
    expression = "resource.name.startsWith(\"${local.training_data_objects}/drift/\")"
  }
}
