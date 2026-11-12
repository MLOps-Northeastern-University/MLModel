terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }

  # The state file is Terraform's record of what it made. Keep it in a bucket so
  # the whole team shares one copy. This bucket must exist before `terraform init`.
  # backend "gcs" {
  #   bucket = "team17-mlops-neu-tfstate"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
