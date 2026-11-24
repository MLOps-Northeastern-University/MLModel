# The Backend's API on Cloud Run, the service the phone app and the care portal call. Its settings
# live here; the Backend's deploy workflow only swaps the image (.github/workflows/deploy.yml there).

locals {
  # What the API reads at startup (app/config.py). A setting left null is not set at all.
  api_env = { for name, value in {
    # A list as JSON, which is how the Backend's settings read a list from the environment; []
    # lets no browser page call the API.
    CORS_ORIGINS = jsonencode(var.careportal_origins)
    # Both default to false already (app/config.py). As environment variables they also win over any
    # .env file, so the deployed API never accepts "dev-" tokens or serves the /dev page (no sign-in).
    DEV_AUTH     = "false"
    DEV_TOOLS    = "false"
    PUSH_ENABLED = var.firebase_project == null ? "false" : "true"
    # Firebase checks sign-ins against this project. On Cloud Run it would otherwise be this one,
    # and every sign-in from a separate Firebase project fails as the wrong audience.
    GOOGLE_CLOUD_PROJECT = var.firebase_project
  } : name => value if value != null }

  # Read from Secret Manager when an instance starts (backend.tf).
  api_secrets = {
    DATABASE_URL = google_secret_manager_secret.backend_database_url.secret_id
    SECRET_KEY   = google_secret_manager_secret.backend_secret_key.secret_id
  }
}

# Made once by Terraform from the hand-pushed image below. After that backend-deploy
# (ci_access.tf) rolls out each new image, and Terraform leaves the image alone (ignore_changes).
resource "google_cloud_run_v2_service" "api" {
  name     = "api"
  location = var.region

  # It holds no data (that's in Cloud SQL), so it can be replaced freely.
  deletion_protection = false

  # Public: the API checks the Firebase sign-in on every request itself. This skips Cloud Run's
  # own check instead of granting allUsers, which an organization policy can block.
  invoker_iam_disabled = true

  # The deploy workflow sets the image (and gcloud its client fields); Terraform keeps the rest.
  # GCP fills in a service-wide scaling block, as on mlflow; the real limits are below.
  lifecycle {
    ignore_changes = [scaling, client, client_version, template[0].containers[0].image]
  }

  template {
    # backend-api (backend.tf): the database and its two secrets, nothing else. A deploy may keep
    # this account because backend-deploy has serviceAccountUser on it (ci_access.tf).
    service_account = google_service_account.backend_api.email

    # Sleeps when unused. One copy: mednova-db (a db-f1-micro) takes about 25 connections, shared
    # with MLflow and the jobs, and SQLAlchemy opens up to 15 per copy (app/db.py).
    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }

    # Opens the private connection to mednova-db at /cloudsql/<connection name>, which
    # DATABASE_URL points at.
    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [google_sql_database_instance.main.connection_name]
      }
    }

    containers {
      # The first image only: the one pushed by hand. Every deploy replaces it.
      image = "us-central1-docker.pkg.dev/${var.project_id}/ml/api:6403d17"

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
        # Billed only while handling requests.
        cpu_idle = true
      }

      dynamic "env" {
        for_each = local.api_env
        content {
          name  = env.key
          value = env.value
        }
      }
      dynamic "env" {
        for_each = local.api_secrets
        content {
          name = env.key
          value_source {
            secret_key_ref {
              secret  = env.value
              version = "latest"
            }
          }
        }
      }

      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }
    }
  }

  # The API needs its permissions and both secrets in place before it first starts.
  depends_on = [
    google_project_iam_member.backend_sql,
    google_secret_manager_secret_iam_member.backend_settings,
    google_secret_manager_secret_version.backend_database_url,
    google_secret_manager_secret_version.backend_secret_key,
  ]
}

# `terraform output BACKEND_API_URL` prints the API's run.app address: the one to point the
# phone app and the care portal at.
output "BACKEND_API_URL" {
  value = google_cloud_run_v2_service.api.uri
}
