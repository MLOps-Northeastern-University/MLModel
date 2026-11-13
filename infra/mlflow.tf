resource "google_service_account" "mlflow" {
  account_id   = "mlflow"
  display_name = "MLflow server (Cloud Run)"

  depends_on = [google_project_service.apis]
}

# Connect to mednova-db for run history. Cloud SQL can only grant this on the whole
# project, not one instance.
resource "google_project_iam_member" "mlflow_sql" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.mlflow.email}"
}

# Plots and reports in models/mlflow/.
resource "google_storage_bucket_iam_member" "mlflow_models" {
  bucket = google_storage_bucket.buckets["models"].name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.mlflow.email}"
}

# The database password, read at start-up.
resource "google_secret_manager_secret_iam_member" "mlflow_db_password" {
  secret_id = google_secret_manager_secret.mlflow_db_password.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.mlflow.email}"
}

# The shared MLflow server: the image from Model_Pipeline/mlflow/, run on Cloud Run as
# the mlflow account. Only accounts given run.invoker can reach it (see the step 9
# access item); everyone else gets 403.
resource "google_cloud_run_v2_service" "mlflow" {
  name     = "mlflow"
  location = var.region

  # It holds no data (that's in Cloud SQL and the bucket), so it can be replaced freely.
  deletion_protection = false

  # GCP fills in a service-wide scaling block of zeros (the defaults) that Terraform would
  # otherwise try to clear on every plan. The real limits are in template.scaling below.
  lifecycle {
    ignore_changes = [scaling]
  }

  template {
    service_account = google_service_account.mlflow.email

    # Sleeps when unused, so idle time is free; one copy is plenty for the team.
    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }

    # Opens the private connection to mednova-db at /cloudsql/<connection name>.
    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [google_sql_database_instance.main.connection_name]
      }
    }

    containers {
      image = "us-central1-docker.pkg.dev/${var.project_id}/ml/mlflow:3.17.2-2"

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
        # Billed only while handling requests.
        cpu_idle = true
      }

      env {
        name  = "CLOUD_SQL_CONNECTION"
        value = google_sql_database_instance.main.connection_name
      }
      env {
        name = "DB_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.mlflow_db_password.secret_id
            version = "latest"
          }
        }
      }

      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }
    }
  }

  # The server needs its permissions and the password in place before it first starts.
  depends_on = [
    google_project_iam_member.mlflow_sql,
    google_storage_bucket_iam_member.mlflow_models,
    google_secret_manager_secret_iam_member.mlflow_db_password,
    google_secret_manager_secret_version.mlflow_db_password,
  ]
}

# Lets CI (the training account) log runs to the server. Teammates already can, through Editor.
resource "google_cloud_run_v2_service_iam_member" "training_mlflow" {
  name     = google_cloud_run_v2_service.mlflow.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.training.email}"
}

output "MLFLOW_TRACKING_URI" {
  value = google_cloud_run_v2_service.mlflow.uri
}
