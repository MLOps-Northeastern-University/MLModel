# Step 2: the buckets.

resource "google_storage_bucket" "buckets" {
  # training-data holds the DVC remote (cops-dvc/); models holds MLflow's files (mlflow/).
  for_each = toset(["training-data", "models"])
  # Bucket names are global across GCP, hence the project prefix: team17-mlops-neu-training-data
  # and team17-mlops-neu-models. Both were made by hand first; imports.tf hands them to Terraform.
  name     = "${var.project_id}-${each.key}"
  location = var.region

  # Access is granted with IAM on the whole bucket (ci_access.tf, mlflow.tf, backend.tf,
  # training_vm_access.tf), never with per-object ACLs. IAM conditions, such as backend.tf's
  # harnet10-only grant, only work on a bucket with this turned on.
  uniform_bucket_level_access = true
  # Nothing in either bucket can be made public, even by a grant to allUsers by mistake: they
  # hold the licensed COPS data and the models.
  public_access_prevention = "enforced"

  # No force_destroy, so Terraform refuses to delete a bucket that still has files in it.
  depends_on = [google_project_service.apis]
}
