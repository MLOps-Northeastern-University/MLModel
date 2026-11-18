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

variable "training_regions" {
  # Where CI may make a training VM, tried in this order until one has the machine free
  # (.github/workflows/model.yml). us-central1 first: the data bucket is there. The others offer
  # T4 and L4 GPUs too; a VM elsewhere copies the data across regions, slower and billed.
  default = ["us-central1", "us-east1", "us-east4", "us-west1"]
}

variable "airflow_users" {
  # One Airflow login on the training VM per teammate (secrets.tf makes the passwords).
  default = ["sam", "ayush", "basim", "dhruv", "tejaswini"]
}

variable "billing_account" {
  # The billing account's ID (Billing -> Account management), for the budget alert. Left empty,
  # there's no budget: setting one needs permission on the billing account.
  default = ""
}

variable "monthly_budget_usd" {
  default = 50
}

variable "budget_alert_email" {
  # Who the budget emails besides the billing account's admins. Empty: only them.
  default = ""
}
