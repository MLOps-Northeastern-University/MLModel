variable "project_id" {
  default = "team17-mlops-neu"
}

variable "region" {
  # One region for everything, where the free tiers are (see the architecture page).
  default = "us-central1"
}

variable "github_repo" {
  default = "MLOps-Northeastern-University/MLModel"
}
