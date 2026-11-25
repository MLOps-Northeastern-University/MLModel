# The settings the other files read. Each has the team's value as its default, so a plain
# `terraform plan` needs no -var flags or .tfvars file.
variable "project_id" {
  # The team project, where everything in this folder is made. The training VMs' project is
  # infra/compute/'s own variable (compute_project), applied separately.
  default = "team17-mlops-neu"
}

variable "region" {
  # One region for everything, where the free tiers are (see the architecture page).
  default = "us-central1"
}

variable "github_repo" {
  # The only repository whose GitHub tokens the mlmodel provider accepts (ci_access.tf), as
  # <org>/<repo>. If the repo is renamed or moved, change this too, or CI can't sign in to GCP.
  default = "MLOps-Northeastern-University/MLModel"
}

variable "backend_repo" {
  # Its deploy workflow logs in as backend-deploy (ci_access.tf).
  default = "MLOps-Northeastern-University/Backend"
}

variable "careportal_origins" {
  # The care portal's addresses, e.g. ["https://<site>.web.app"]: the only web pages that may call
  # the Backend's API from a browser (its CORS_ORIGINS). Empty until the portal is hosted.
  type    = list(string)
  default = []
}

variable "firebase_project" {
  # The Firebase project staff and patients sign in with, and the pushes go through: mednova-dhruv,
  # its own Google Cloud project. Null turns sign-in to the API and pushes to the phone off. The
  # Backend's accounts get their roles there by hand (Backend docs/DEPLOY.md, "Firebase is in its
  # own project"), since Terraform here only signs in to this project.
  type    = string
  default = "mednova-dhruv"
}

variable "firebase_web_api_key_added" {
  # True once backend-firebase-web-api-key (backend.tf) has its value, added by hand:
  #   printf %s "<the web app's key>" | gcloud secrets versions add backend-firebase-web-api-key --data-file=-
  # Only then does the API read it: Cloud Run won't start a service whose secret has no version.
  type    = bool
  default = false
}

variable "scoring_paused" {
  # Pauses score-hours' and check-drift's schedules (backend_run.tf). Both stop at the start without
  # harnet10's weights in the training-data bucket's harnet10/. They're there since 2026-11-24.
  type    = bool
  default = false
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
