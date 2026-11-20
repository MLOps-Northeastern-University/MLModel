variable "compute_project" {
  # Where the training VMs run: Sam's project, for its GPU quota and credits. To move them to the
  # team project later, set this to team17-mlops-neu (and training_vm_email in ../ to match).
  default = "finetuning-499422"
}

variable "region" {
  # The team's data bucket is in us-central1: VMs there copy it for free.
  default = "us-central1"
}

variable "training_regions" {
  # Where CI may make a training VM, tried in this order until one has the machine free. Each has
  # T4 and L4 GPUs; a VM outside us-central1 copies the data across regions, slower and billed.
  default = ["us-central1", "us-east1", "us-east4", "us-west1"]
}

variable "ci_service_account" {
  # CI's account in the team project (../ci_access.tf): it makes and deletes the VMs here.
  default = "training@team17-mlops-neu.iam.gserviceaccount.com"
}

variable "billing_account" {
  # This project's billing account ID (Billing -> Account management), for the budget alert.
  # Empty: no budget. Making one needs permission on the billing account.
  default = ""
}

variable "monthly_budget_usd" {
  default = 50
}

variable "budget_alert_email" {
  # Who the budget emails besides the billing account's admins. Empty: only them.
  default = ""
}
