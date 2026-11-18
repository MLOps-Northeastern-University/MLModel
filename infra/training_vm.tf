# The training VMs. CI makes one for each training run and deletes it after (.github/workflows/
# model.yml, airflow/vm/), so there's no VM here: only what every run's VM shares. Who it runs as,
# what it may touch, and a network nothing can connect into.

# ---- who it runs as, and what it may touch -----------------------------------------------

resource "google_service_account" "training_vm" {
  account_id   = "training-vm"
  display_name = "Training VM (one per run, made by CI)"

  depends_on = [google_project_service.apis]
}

# Read and write both buckets: in training-data, the DVC data, harnet10's weights (harnet10/) and
# the feature cache (cache/), kept between runs; in models, the registry.
resource "google_storage_bucket_iam_member" "training_vm_buckets" {
  for_each = google_storage_bucket.buckets
  bucket   = each.value.name
  role     = "roles/storage.objectUser"
  member   = "serviceAccount:${google_service_account.training_vm.email}"
}

# Log each run to the shared MLflow, which lets in only callers with run.invoker.
resource "google_cloud_run_v2_service_iam_member" "training_vm_mlflow" {
  name     = google_cloud_run_v2_service.mlflow.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.training_vm.email}"
}

# Pull the Airflow and pipeline images CI pushed.
resource "google_artifact_registry_repository_iam_member" "training_vm_ml" {
  repository = google_artifact_registry_repository.ml.name
  location   = var.region
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.training_vm.email}"
}

# Its logs and metrics in Cloud Logging and Monitoring.
resource "google_project_iam_member" "training_vm_observability" {
  for_each = toset(["roles/logging.logWriter", "roles/monitoring.metricWriter"])
  project  = var.project_id
  role     = each.key
  member   = "serviceAccount:${google_service_account.training_vm.email}"
}

# ---- a network nothing can connect into ------------------------------------------------

# Its own network, not the project's default one, whose rules let anyone try SSH. A new network
# lets nothing in; the only rule below is SSH through Google's IAP, which needs a Google login
# with access to the project. The VMs reach out (images, GCS, MLflow, the Cloudflare tunnel that
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
