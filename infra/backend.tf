# The Backend's own database on mednova-db (database.tf), next to MLflow's, and the two settings it
# can't start without. Both are kept in Secret Manager, so they're never in code, config or GitHub;
# the Backend's Cloud Run service and scoring jobs read them as DATABASE_URL and SECRET_KEY.

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
