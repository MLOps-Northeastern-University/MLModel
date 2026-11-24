# The Backend on Cloud Run: the API, the service the phone app and the care portal call, and the
# jobs that run from the same image, with their schedules. Their settings live here; the Backend's
# deploy workflow only swaps the image (.github/workflows/deploy.yml there).

locals {
  # The image Terraform first makes the service and the jobs with, pushed by hand (Backend commit
  # 2e6c76c). After that backend-deploy (ci_access.tf) rolls out each new image, and Terraform
  # leaves the image alone (ignore_changes).
  backend_image = "us-central1-docker.pkg.dev/${var.project_id}/ml/api:2e6c76c"

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

  # Read from Secret Manager when an instance or a job starts (backend.tf). The jobs need both too:
  # the Backend's settings won't load without SECRET_KEY.
  secret_env = {
    DATABASE_URL = google_secret_manager_secret.backend_database_url.secret_id
    SECRET_KEY   = google_secret_manager_secret.backend_secret_key.secret_id
  }
}

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
      # The first image only. Every deploy replaces it.
      image = local.backend_image

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
        for_each = local.secret_env
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

# ---- the jobs and their schedules ---------------------------------------------------------------

locals {
  # Each runs one command from the API's image to the end, then stops (Backend docs/DEPLOY.md).
  # The deploy workflow updates all five to each new image, so none may be missing. Schedules are UTC.
  backend_jobs = {
    # Before each new version, run by the deploy workflow rather than a schedule. No retry: a
    # migration that failed halfway needs a look first.
    migrate = {
      command  = ["alembic", "upgrade", "head"]
      cpu      = "1"
      memory   = "512Mi"
      timeout  = "600s"
      retries  = 0
      schedule = null
      paused   = null
    }
    # Scores each patient's finished hours with the released model. Memory because /tmp is memory on
    # Cloud Run, and it holds the bundle, harnet10 and harnet10's feature cache. At :20, once each
    # hour's 15 minutes for late uploads are over.
    score-hours = {
      command  = ["python", "-m", "scripts.score_hours"]
      cpu      = "2"
      memory   = "4Gi"
      timeout  = "1800s"
      retries  = 1
      schedule = "20 * * * *"
      paused   = var.scoring_paused
    }
    # Yesterday's model metrics. Before check-drift: it replaces all of the night's metrics,
    # check-drift's psi_p_off among them, so running it after would wipe the drift number.
    model-metrics = {
      command  = ["python", "-m", "scripts.compute_model_metrics"]
      cpu      = "1"
      memory   = "2Gi"
      timeout  = "1800s"
      retries  = 1
      schedule = "0 2 * * *"
      paused   = false
    }
    # Last night's drift; a report in training-data/drift/ when a retrain is due, which MLModel's
    # drift.yml turns into an issue.
    check-drift = {
      command  = ["python", "-m", "scripts.check_drift"]
      cpu      = "1"
      memory   = "2Gi"
      timeout  = "1800s"
      retries  = 1
      schedule = "30 2 * * *"
      paused   = var.scoring_paused
    }
    # Ends monitoring whose window ran out and deletes data kept past its retention.
    end-and-purge = {
      command  = ["python", "-m", "scripts.end_and_purge"]
      cpu      = "1"
      memory   = "1Gi"
      timeout  = "900s"
      retries  = 1
      schedule = "0 3 * * *"
      paused   = false
    }
  }

  scheduled_jobs = { for name, job in local.backend_jobs : name => job if job.schedule != null }

  # Where the registry, harnet10 and the drift reports are: this project's buckets, not the defaults
  # in app/config.py, which name team17-mlops-neu, so a copy in another project never reads ours.
  job_env = { for name, value in {
    MODEL_REGISTRY = "gs://${google_storage_bucket.buckets["models"].name}"
    HARNET_DIR     = "gs://${google_storage_bucket.buckets["training-data"].name}/harnet10"
    DRIFT_REPORTS  = "gs://${google_storage_bucket.buckets["training-data"].name}/drift"
    # end-and-purge tells the phone its monitoring ended, as the API's pushes do.
    PUSH_ENABLED         = local.api_env["PUSH_ENABLED"]
    GOOGLE_CLOUD_PROJECT = var.firebase_project
  } : name => value if value != null }
}

resource "google_cloud_run_v2_job" "backend" {
  for_each = local.backend_jobs

  name     = each.key
  location = var.region

  # They hold no data, like the API.
  deletion_protection = false

  # As on the API: the deploy workflow sets the image (and gcloud its client fields).
  lifecycle {
    ignore_changes = [client, client_version, template[0].template[0].containers[0].image]
  }

  template {
    template {
      # backend-scoring (backend.tf): the database, its two secrets, the registry, harnet10 and the
      # drift reports. backend-deploy may keep it on an update (ci_access.tf).
      service_account = google_service_account.backend_scoring.email
      timeout         = each.value.timeout
      max_retries     = each.value.retries

      volumes {
        name = "cloudsql"
        cloud_sql_instance {
          instances = [google_sql_database_instance.main.connection_name]
        }
      }

      containers {
        # The first image only. Every deploy replaces it. The command replaces the image's own
        # (uvicorn, the API).
        image   = local.backend_image
        command = each.value.command

        resources {
          limits = {
            cpu    = each.value.cpu
            memory = each.value.memory
          }
        }

        dynamic "env" {
          for_each = local.job_env
          content {
            name  = env.key
            value = env.value
          }
        }
        dynamic "env" {
          for_each = local.secret_env
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
  }

  depends_on = [
    google_project_iam_member.backend_sql,
    google_secret_manager_secret_iam_member.backend_settings,
    google_secret_manager_secret_version.backend_database_url,
    google_secret_manager_secret_version.backend_secret_key,
  ]
}

# Cloud Scheduler signs in as this account to start the jobs. It can do nothing else: no database,
# secrets or buckets.
resource "google_service_account" "backend_scheduler" {
  account_id   = "backend-scheduler"
  display_name = "Backend job schedules (Cloud Scheduler)"

  depends_on = [google_project_service.apis]
}

# Starting a job (run.jobs.run) is in run.invoker. On each scheduled job only, so not on migrate.
resource "google_cloud_run_v2_job_iam_member" "scheduler_runs" {
  for_each = local.scheduled_jobs

  name     = google_cloud_run_v2_job.backend[each.key].name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.backend_scheduler.email}"
}

# Each schedule calls Cloud Run's "run this job" with a token for backend-scheduler. A missed run is
# made up by the next one: score-hours looks back 48 hours, and the nightly ones take a date by hand.
resource "google_cloud_scheduler_job" "backend" {
  for_each = local.scheduled_jobs

  name      = each.key
  region    = var.region
  schedule  = each.value.schedule
  time_zone = "Etc/UTC"
  paused    = each.value.paused

  http_target {
    http_method = "POST"
    uri         = "https://run.googleapis.com/v2/${google_cloud_run_v2_job.backend[each.key].id}:run"
    oauth_token {
      service_account_email = google_service_account.backend_scheduler.email
    }
  }

  depends_on = [google_cloud_run_v2_job_iam_member.scheduler_runs]
}
