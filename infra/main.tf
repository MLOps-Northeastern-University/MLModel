terraform {
  required_providers {
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
    bucket = "team17-mlops-neu-tfstate"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
