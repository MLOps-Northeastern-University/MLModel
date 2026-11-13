resource "google_sql_database_instance" "main" {
  name             = "mednova-db"
  database_version = "POSTGRES_16"
  region           = var.region

  settings {
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

resource "google_secret_manager_secret_version" "mlflow_db_password" {
  secret      = google_secret_manager_secret.mlflow_db_password.id
  secret_data = random_password.mlflow_db.result
}
