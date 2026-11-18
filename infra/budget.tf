# A monthly budget for the project, so a training VM left running (or a GPU run that hangs) shows
# up in an email long before it uses up the credits. Made only when var.billing_account is set:
# a budget lives on the billing account, which needs its own permission (Billing Account
# Administrator, or Costs Manager).

data "google_project" "this" {}

# The extra address the budget emails, besides the billing account's admins.
resource "google_monitoring_notification_channel" "budget" {
  count        = var.billing_account != "" && var.budget_alert_email != "" ? 1 : 0
  display_name = "Budget alerts"
  type         = "email"
  labels = {
    email_address = var.budget_alert_email
  }

  depends_on = [google_project_service.apis]
}

resource "google_billing_budget" "project" {
  count           = var.billing_account != "" ? 1 : 0
  billing_account = var.billing_account
  display_name    = "${var.project_id} monthly"

  budget_filter {
    projects = ["projects/${data.google_project.this.number}"]
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.monthly_budget_usd)
    }
  }

  # An email at half the budget, at 90%, and when it's all spent.
  threshold_rules {
    threshold_percent = 0.5
  }
  threshold_rules {
    threshold_percent = 0.9
  }
  threshold_rules {
    threshold_percent = 1.0
  }

  all_updates_rule {
    monitoring_notification_channels = google_monitoring_notification_channel.budget[*].id
    # The billing account's admins get the emails too.
    disable_default_iam_recipients = false
  }

  depends_on = [google_project_service.apis]
}
