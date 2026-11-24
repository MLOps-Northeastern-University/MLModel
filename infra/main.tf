terraform {
  required_providers {
    # ~> 6.0 takes any 6.x release but never 7.0, whose breaking changes need a deliberate upgrade.
    # .terraform.lock.hcl (committed) pins the exact build, so every teammate's plan uses the same one.
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
    # Generates the database password (database.tf).
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  # The state file is Terraform's record of what it made. It lives in this bucket so
  # the whole team shares one copy, with old versions kept. The bucket was made by hand
  # (Terraform can't create the bucket it keeps its own state in).
  backend "gcs" {
    # No prefix: this folder's state is default.tfstate at the bucket's root, and infra/compute/
    # keeps its own under compute/, so neither apply can touch the other's record.
    bucket = "team17-mlops-neu-tfstate"
  }
}

# Every resource is made in the team project and region unless it names its own. There's no key
# file: Terraform works as whoever runs it, through `gcloud auth application-default login`.
provider "google" {
  project = var.project_id
  region  = var.region
}
