# Step 2: the APIs.

# An API has to be on before anything of its kind can be made, which is why the other files'
# resources depend_on this one. Compute isn't listed: the training VMs run in the compute
# project, whose APIs infra/compute/training_vm.tf turns on.
resource "google_project_service" "apis" {
  for_each = toset([
    "storage",          # buckets
    "artifactregistry", # Docker images and model bundles
    "run",              # the shared MLflow server
    "sqladmin",         # Cloud SQL, for MLflow's run history
    "secretmanager",    # the MLflow database password
    # iam, iamcredentials and sts together make Workload Identity Federation work: sts swaps
    # GitHub's ID token for a Google one, and iamcredentials turns that into a short-lived token
    # for the training or backend-deploy account (ci_access.tf).
    "iam",            # service accounts and the WIF pool
    "iamcredentials", # short-lived logins for the service account
    "sts",            # checks GitHub's ID token
    "cloudscheduler", # starts the Backend's scoring jobs on a timer
    "logging",        # the Backend's and the jobs' logs
    "monitoring",     # uptime checks, alert emails and the dashboard
  ])
  service = "${each.key}.googleapis.com"

  # Removing a line here shouldn't switch the API off under the Backend.
  disable_on_destroy = false
}
