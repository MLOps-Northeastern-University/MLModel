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

variable "training_vm_email" {
  # The training VMs' account, made in their own project by compute/ (its output training_vm_email).
  # This project lets it use the buckets, MLflow, the images and its secrets.
  default = "training-vm@finetuning-499422.iam.gserviceaccount.com"
}

variable "airflow_users" {
  # One Airflow login on the training VM per teammate (secrets.tf makes the passwords).
  default = ["sam", "ayush", "basim", "dhruv", "tejaswini"]
}
