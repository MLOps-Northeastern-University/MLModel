# What every training VM shares. There's no VM here: CI makes one per run and deletes it after.

resource "google_project_service" "apis" {
  for_each = toset([
    "compute",        # the VMs
    "iam",            # their service account
    "logging",        # their logs
    "monitoring",     # their metrics, and the budget's alert emails
    "billingbudgets", # the budget alert (budget.tf)
  ])
  service            = "${each.key}.googleapis.com"
  disable_on_destroy = false
}

# ---- who it runs as ----------------------------------------------------------------------

# What it may do in the team project (its buckets, MLflow, images, secrets) is granted there,
# in ../training_vm_access.tf.
resource "google_service_account" "training_vm" {
  account_id   = "training-vm"
  display_name = "Training VM (one per run, made by CI)"

  depends_on = [google_project_service.apis]
}

# Its logs and metrics in this project's Cloud Logging and Monitoring.
resource "google_project_iam_member" "training_vm_observability" {
  for_each = toset(["roles/logging.logWriter", "roles/monitoring.metricWriter"])
  project  = var.compute_project
  role     = each.key
  member   = "serviceAccount:${google_service_account.training_vm.email}"
}

# Delete itself when its run ends (airflow/vm/startup.sh), so a run longer than CI's six-hour job
# still cleans up. Only instances named mednova-run-*, so it can't touch any other VM here.
resource "google_project_iam_custom_role" "delete_training_vm" {
  role_id     = "deleteTrainingVm"
  title       = "Delete a training VM"
  permissions = ["compute.instances.delete"]
}

resource "google_project_iam_member" "training_vm_deletes_itself" {
  project = var.compute_project
  role    = google_project_iam_custom_role.delete_training_vm.name
  member  = "serviceAccount:${google_service_account.training_vm.email}"

  condition {
    title      = "training VMs only"
    expression = "resource.name.extract(\"/instances/{name}\").startsWith(\"mednova-run-\")"
  }
}

# ---- who may make the VMs: CI ------------------------------------------------------------

# Create, start and delete the VM of each training run.
resource "google_project_iam_member" "ci_instances" {
  project = var.compute_project
  role    = "roles/compute.instanceAdmin.v1"
  member  = "serviceAccount:${var.ci_service_account}"
}

# Make that VM run as training-vm: a VM can only be given an account its maker may act as.
resource "google_service_account_iam_member" "ci_acts_as_vm" {
  service_account_id = google_service_account.training_vm.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${var.ci_service_account}"
}

# ---- a network nothing can connect into ------------------------------------------------

# Its own network, not the project's default one, whose rules let anyone try SSH. A new network
# lets nothing in; the only rule below is SSH through Google's IAP, which needs a Google login
# with access to this project. The VMs reach out (images, GCS, MLflow, the Cloudflare tunnel that
# serves the Airflow UI), so they keep an external address, but no port is open on it.
resource "google_compute_network" "training" {
  name                    = "training"
  auto_create_subnetworks = false

  depends_on = [google_project_service.apis]
}

# One subnet in each region CI may use.
resource "google_compute_subnetwork" "training" {
  for_each      = { for i, region in var.training_regions : region => i }
  name          = "training-${each.key}"
  region        = each.key
  network       = google_compute_network.training.id
  ip_cidr_range = "10.${10 + each.value}.0.0/20"
  # Google's APIs (GCS, Artifact Registry) over Google's own network.
  private_ip_google_access = true
}

# SSH, for debugging a run, only from IAP's addresses: gcloud compute ssh --tunnel-through-iap.
resource "google_compute_firewall" "training_iap_ssh" {
  name          = "training-allow-iap-ssh"
  network       = google_compute_network.training.name
  direction     = "INGRESS"
  source_ranges = ["35.235.240.0/20"]
  target_tags   = ["training-vm"]

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

output "training_vm_email" {
  # ../ grants this account access to the team's resources (variable training_vm_email there).
  value = google_service_account.training_vm.email
}
