# What the training VM reads from Secret Manager at start-up (airflow/vm/), so no password is in a
# file, an image or the repo.

# ---- the Airflow logins: one per teammate -------------------------------------------------

# Generated here, so each is long and random. They're in Terraform's state too, which lives in the
# private tfstate bucket. To read yours:
#   gcloud secrets versions access latest --secret airflow-users | python -c "import json,sys; print(json.load(sys.stdin)['sam'])"
resource "random_password" "airflow_user" {
  for_each = toset(var.airflow_users)
  length   = 24
  special  = false
}

resource "google_secret_manager_secret" "airflow_users" {
  secret_id = "airflow-users"
  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

# {"sam": "<password>", ...}
resource "google_secret_manager_secret_version" "airflow_users" {
  secret      = google_secret_manager_secret.airflow_users.id
  secret_data = jsonencode({ for user, password in random_password.airflow_user : user => password.result })
}

# ---- the alerts' settings: slots only, filled in by hand ----------------------------------

# Their values aren't in Terraform, so a real password never reaches the code or its state.
# Add one with:
#   echo -n '{"to": "...", "host": "smtp.gmail.com", "port": 587, "user": "...", "password": "..."}' \
#     | gcloud secrets versions add alert-smtp --data-file=-
#   echo -n 'https://hooks.slack.com/...' | gcloud secrets versions add alert-slack-webhook --data-file=-
resource "google_secret_manager_secret" "alerts" {
  for_each  = toset(["alert-smtp", "alert-slack-webhook"])
  secret_id = each.key
  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

# ---- who may read them: the training VMs (their account is made in compute/) ------------

resource "google_secret_manager_secret_iam_member" "training_vm_reads" {
  for_each = merge(
    { "airflow-users" = google_secret_manager_secret.airflow_users.id },
    { for name, secret in google_secret_manager_secret.alerts : name => secret.id },
  )
  secret_id = each.value
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${var.training_vm_email}"
}
