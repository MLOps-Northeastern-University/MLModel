# Step 2: the APIs.

resource "google_project_service" "apis" {
  for_each = toset([
    "storage",          # buckets
    "artifactregistry", # Docker images and model bundles
    "run",              # the shared MLflow server
    "sqladmin",         # Cloud SQL, for MLflow's run history
    "secretmanager",    # the MLflow database password
    "iam",              # service accounts and the WIF pool
    "iamcredentials",   # short-lived logins for the service account
    "sts",              # checks GitHub's ID token
    "cloudscheduler",   # starts the Backend's scoring jobs on a timer
    "logging",          # the Backend's and the jobs' logs
    "monitoring",       # uptime checks, alert emails and the dashboard
  ])
  service = "${each.key}.googleapis.com"

  # Removing a line here shouldn't switch the API off under the Backend.
  disable_on_destroy = false
}
