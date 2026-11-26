# Who hears when the deployed Backend breaks: an email to each teammate when the API is down or
# failing, a job fails, or scoring stops. Only a failure sends one; a run that succeeds, or an alert
# that clears, sends nothing. Then the dashboard to watch it on.

# ---- who gets the emails ------------------------------------------------------------------------

# The team's addresses, comma-separated like ALERT_EMAIL. In Secret Manager rather than here, since
# this repo is public. Added by hand (variables.tf, alert_emails_added).
resource "google_secret_manager_secret" "alert_emails" {
  secret_id = "alert-emails"
  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

data "google_secret_manager_secret_version" "alert_emails" {
  count  = var.alert_emails_added ? 1 : 0
  secret = google_secret_manager_secret.alert_emails.secret_id
}

locals {
  alert_emails = var.alert_emails_added ? [
    for address in split(",", data.google_secret_manager_secret_version.alert_emails[0].secret_data) :
    trimspace(address) if trimspace(address) != ""
  ] : []
}

# One channel per address. Google sends these itself, so they don't need the SMTP password.
resource "google_monitoring_notification_channel" "email" {
  # Numbered, not named after the address, which would put it in the plan.
  count        = nonsensitive(length(local.alert_emails))
  display_name = "Team alert email ${count.index + 1}"
  type         = "email"
  labels = {
    email_address = local.alert_emails[count.index]
  }
}

# ---- is the API up ------------------------------------------------------------------------------

# Google calls /health from three US regions every 15 minutes. Not more often: each call wakes the
# API, which scales to zero. /health, not /healthz: Cloud Run keeps some paths ending in z for itself.
resource "google_monitoring_uptime_check_config" "api" {
  display_name     = "api /health"
  period           = "900s"
  timeout          = "30s" # time for a cold start
  selected_regions = ["USA_IOWA", "USA_OREGON", "USA_VIRGINIA"]

  http_check {
    path         = "/health"
    port         = 443
    use_ssl      = true
    validate_ssl = true
  }

  monitored_resource {
    type = "uptime_url"
    labels = {
      project_id = var.project_id
      host       = trimprefix(google_cloud_run_v2_service.api.uri, "https://")
    }
  }

  depends_on = [google_project_service.apis]
}

# ---- the alerts ---------------------------------------------------------------------------------
# Each emails when it opens, and not when it closes (notification_prompts has no "CLOSED"). A job
# whose first try fails but whose retry succeeds counts as succeeded, so it sends nothing either.

resource "google_monitoring_alert_policy" "api_down" {
  display_name          = "The API is down"
  combiner              = "OR"
  notification_channels = google_monitoring_notification_channel.email[*].id

  conditions {
    display_name = "/health fails from 2 of the 3 regions"
    condition_threshold {
      filter          = "metric.type=\"monitoring.googleapis.com/uptime_check/check_passed\" AND metric.label.check_id=\"${google_monitoring_uptime_check_config.api.uptime_check_id}\" AND resource.type=\"uptime_url\""
      comparison      = "COMPARISON_GT"
      threshold_value = 1
      duration        = "0s"
      aggregations {
        # Longer than the check's 15 minutes, so each window holds a result.
        alignment_period     = "1200s"
        per_series_aligner   = "ALIGN_NEXT_OLDER"
        cross_series_reducer = "REDUCE_COUNT_FALSE"
        group_by_fields      = ["resource.label.*"]
      }
    }
  }

  alert_strategy {
    notification_prompts = ["OPENED"]
  }

  documentation {
    subject   = "MedNOVA: the API is down"
    mime_type = "text/markdown"
    content   = <<-EOT
      The API at ${google_cloud_run_v2_service.api.uri}/health stopped answering. The phone app and the care portal can't reach the Backend.

      Its logs, in Logs Explorer: `resource.type="cloud_run_revision" resource.labels.service_name="api" severity>=ERROR`.
      A bad new image: roll back to the last revision that worked (Cloud Run → api → Revisions).
    EOT
  }
}

resource "google_monitoring_alert_policy" "api_errors" {
  display_name          = "The API is failing requests"
  combiner              = "OR"
  notification_channels = google_monitoring_notification_channel.email[*].id

  conditions {
    display_name = "5 or more 5xx answers in 5 minutes"
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.label.service_name=\"${google_cloud_run_v2_service.api.name}\" AND metric.type=\"run.googleapis.com/request_count\" AND metric.label.response_code_class=\"5xx\""
      comparison      = "COMPARISON_GT"
      threshold_value = 4
      duration        = "60s"
      # No 5xx at all sends no data, which closes the alert rather than holding it open.
      evaluation_missing_data = "EVALUATION_MISSING_DATA_INACTIVE"
      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_SUM"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }

  alert_strategy {
    notification_prompts = ["OPENED"]
  }

  documentation {
    subject   = "MedNOVA: the API is failing requests"
    mime_type = "text/markdown"
    content   = <<-EOT
      The API answered 5 or more requests with a 5xx in 5 minutes. It's up, but something behind it is failing, often the database.

      Its logs, in Logs Explorer: `resource.type="cloud_run_revision" resource.labels.service_name="api" severity>=ERROR`.
    EOT
  }
}

resource "google_monitoring_alert_policy" "job_failed" {
  display_name          = "A Backend job failed"
  combiner              = "OR"
  notification_channels = google_monitoring_notification_channel.email[*].id

  conditions {
    display_name = "A job's run failed"
    condition_threshold {
      # Any of the five (backend_run.tf). An execution fails only once its retry has failed too.
      filter          = "resource.type=\"cloud_run_job\" AND metric.type=\"run.googleapis.com/job/completed_execution_count\" AND metric.label.result=\"failed\""
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      # A failed run stays in the 5-minute window below for 5 minutes, so it's over this for long
      # enough. Then it closes, so the next failed run emails again.
      duration                = "60s"
      evaluation_missing_data = "EVALUATION_MISSING_DATA_INACTIVE"
      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_SUM"
        cross_series_reducer = "REDUCE_SUM"
        # One alert per job, which the email names.
        group_by_fields = ["resource.label.job_name"]
      }
    }
  }

  alert_strategy {
    notification_prompts = ["OPENED"]
  }

  documentation {
    # $${...} is filled in by Cloud Monitoring, not Terraform.
    subject   = "MedNOVA: the $${resource.label.job_name} job failed"
    mime_type = "text/markdown"
    content   = <<-EOT
      The Cloud Run job `$${resource.label.job_name}` failed, on its retry too.

      Its logs, in Logs Explorer: `resource.type="cloud_run_job" resource.labels.job_name="$${resource.label.job_name}"`.
      score-hours makes up missed hours on its next run (it looks back 48 hours); the nightly jobs need a rerun for that night.
    EOT
  }
}

resource "google_monitoring_alert_policy" "scoring_stopped" {
  display_name          = "Scoring stopped"
  combiner              = "OR"
  notification_channels = google_monitoring_notification_channel.email[*].id

  # Paused on purpose (variables.tf, scoring_paused): no alert.
  enabled = !var.scoring_paused

  conditions {
    display_name = "No successful score-hours for 3 hours"
    # Catches what job_failed can't: runs that never start (a paused or deleted schedule) or never end.
    condition_absent {
      filter   = "resource.type=\"cloud_run_job\" AND resource.label.job_name=\"score-hours\" AND metric.type=\"run.googleapis.com/job/completed_execution_count\" AND metric.label.result=\"succeeded\""
      duration = "10800s"
      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_SUM"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }

  alert_strategy {
    notification_prompts = ["OPENED"]
  }

  documentation {
    subject   = "MedNOVA: scoring stopped"
    mime_type = "text/markdown"
    content   = <<-EOT
      score-hours hasn't finished a run in 3 hours, so no new hours are being scored.

      Check that its schedule is on (Cloud Scheduler → score-hours) and the job's recent runs (Cloud Run → Jobs → score-hours).
      Its logs, in Logs Explorer: `resource.type="cloud_run_job" resource.labels.job_name="score-hours"`.
    EOT
  }
}

# ---- the dashboard ------------------------------------------------------------------------------

locals {
  api_filter = "resource.type=\"cloud_run_revision\" AND resource.label.service_name=\"${google_cloud_run_v2_service.api.name}\""

  # Each tile is a chart of one or more metric lines, with its own aggregation.
  dashboard_tiles = [
    {
      title = "API requests per second, by answer"
      lines = [{
        filter = "${local.api_filter} AND metric.type=\"run.googleapis.com/request_count\""
        aggregation = {
          alignmentPeriod    = "60s"
          perSeriesAligner   = "ALIGN_RATE"
          crossSeriesReducer = "REDUCE_SUM"
          groupByFields      = ["metric.label.response_code_class"]
        }
      }]
    },
    {
      title = "API latency, p50 and p95"
      lines = [for percentile in ["50", "95"] : {
        filter = "${local.api_filter} AND metric.type=\"run.googleapis.com/request_latencies\""
        aggregation = {
          alignmentPeriod    = "60s"
          perSeriesAligner   = "ALIGN_DELTA"
          crossSeriesReducer = "REDUCE_PERCENTILE_${percentile}"
        }
      }]
    },
    {
      title = "API 5xx answers"
      lines = [{
        filter = "${local.api_filter} AND metric.type=\"run.googleapis.com/request_count\" AND metric.label.response_code_class=\"5xx\""
        aggregation = {
          alignmentPeriod    = "300s"
          perSeriesAligner   = "ALIGN_SUM"
          crossSeriesReducer = "REDUCE_SUM"
        }
      }]
    },
    {
      title = "API instances"
      lines = [{
        filter = "${local.api_filter} AND metric.type=\"run.googleapis.com/container/instance_count\""
        aggregation = {
          alignmentPeriod    = "60s"
          perSeriesAligner   = "ALIGN_MAX"
          crossSeriesReducer = "REDUCE_SUM"
          groupByFields      = ["metric.label.state"]
        }
      }]
    },
    {
      title = "Job runs per hour, by job and result"
      lines = [{
        filter = "resource.type=\"cloud_run_job\" AND metric.type=\"run.googleapis.com/job/completed_execution_count\""
        aggregation = {
          alignmentPeriod    = "3600s"
          perSeriesAligner   = "ALIGN_SUM"
          crossSeriesReducer = "REDUCE_SUM"
          groupByFields      = ["resource.label.job_name", "metric.label.result"]
        }
      }]
    },
  ]
}

resource "google_monitoring_dashboard" "backend" {
  # Two tiles a row, each half of the 12 columns.
  dashboard_json = jsonencode({
    displayName = "MedNOVA Backend"
    mosaicLayout = {
      columns = 12
      # A position of 0 is left out: the API drops it, and Terraform would then see a change.
      tiles = [for i, tile in local.dashboard_tiles : merge(
        i % 2 == 0 ? {} : { xPos = 6 },
        i < 2 ? {} : { yPos = floor(i / 2) * 4 },
        {
          width  = 6
          height = 4
          widget = {
            title = tile.title
            xyChart = {
              dataSets = [for line in tile.lines : {
                plotType   = "LINE"
                targetAxis = "Y1"
                timeSeriesQuery = {
                  timeSeriesFilter = line
                }
              }]
            }
          }
      })]
    }
  })

  depends_on = [google_project_service.apis]
}
