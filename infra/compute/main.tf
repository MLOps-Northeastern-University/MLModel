# The training VMs' project: where CI makes one VM for each training run (.github/workflows/
# model.yml, airflow/vm/). Its own Terraform, apart from the team project's (../), because it's a
# different project with a different owner: whoever owns it applies this, from this folder.
#
#   terraform init
#   terraform plan     # what would change
#   terraform apply
#
# Apply this first: ../ grants the account made here access to the team's buckets, MLflow, images
# and secrets.

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
  # The team's state bucket, under its own prefix, so the two never touch each other's records.
  backend "gcs" {
    bucket = "team17-mlops-neu-tfstate"
    prefix = "compute"
  }
}

provider "google" {
  project = var.compute_project
  region  = var.region
}
