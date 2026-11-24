# One small Postgres for the team: MLflow's run history (database mlflow, below) and the
# Backend's data (database mednova, backend.tf). Cloud Run reaches it through its Cloud SQL
# volume, which signs in with the service account's cloudsql.client role, so no network is let in.
resource "google_sql_database_instance" "main" {
  name             = "mednova-db"
  database_version = "POSTGRES_16"
  region           = var.region

  settings {
    # Enterprise, because Enterprise Plus has no shared-core machines. db-f1-micro is the cheapest
    # tier and takes about 25 connections in all (backend_run.tf keeps the API to one copy for it).
    # ZONAL: no standby copy in a second zone, which would double the price.
    edition           = "ENTERPRISE"
    tier              = "db-f1-micro"
    availability_type = "ZONAL"
    disk_type         = "PD_SSD"
    # 10 GB is Cloud SQL's smallest disk; it may grow on its own, but not past 20.
    disk_size             = 10
    disk_autoresize_limit = 20

    # Stops the instance being deleted from the console or gcloud too.
    deletion_protection_enabled = true

    backup_configuration {
      # A backup each day, of which Cloud SQL keeps the last 7 by default: the way back if MLflow's
      # history or the Backend's data is lost.
      enabled = true
    }
  }

  # Stops `terraform destroy` or a bad change from deleting it, and its data.
  deletion_protection = true

  depends_on = [google_project_service.apis]
}

resource "google_sql_database" "mlflow" {
  name     = "mlflow"
  instance = google_sql_database_instance.main.name
}

# Letters and digits only, so the password can go straight into MLflow's database URL.
resource "random_password" "mlflow_db" {
  length  = 32
  special = false
}

# A built-in Postgres login with a password, not an IAM one, because MLflow's database URL
# needs a user and password (mlflow/Dockerfile builds it from DB_PASSWORD).
resource "google_sql_user" "mlflow" {
  name     = "mlflow"
  instance = google_sql_database_instance.main.name
  password = random_password.mlflow_db.result
}

# The MLflow server reads the password from here at start-up, so it's never in code or config.
resource "google_secret_manager_secret" "mlflow_db_password" {
  secret_id = "mlflow-db-password"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

# The password itself, as the secret's version. Like every random_password, it's also in
# Terraform's state, which is one reason the state stays in the private tfstate bucket.
resource "google_secret_manager_secret_version" "mlflow_db_password" {
  secret      = google_secret_manager_secret.mlflow_db_password.id
  secret_data = random_password.mlflow_db.result
}
