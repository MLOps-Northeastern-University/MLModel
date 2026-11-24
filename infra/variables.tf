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
  # The Firebase project staff and patients sign in with. Until it exists (null), nobody can sign
  # in to the API and pushes to the phone stay off.
  type    = string
  default = null
}

variable "scoring_paused" {
  # Pauses score-hours' and check-drift's schedules (backend_run.tf). Both stop at the start without
  # harnet10's weights in the training-data bucket's harnet10/. Set false once they're there.
  type    = bool
  default = true
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
