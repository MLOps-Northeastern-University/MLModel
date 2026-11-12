# The buckets were made by hand with gcloud, so Terraform takes them over instead of
# creating them again. Only works if these names exist exactly (check with
# `gcloud storage ls --project=team17-mlops-neu`). Once they're in the state this
# file does nothing, so it can be deleted after the first apply.

import {
  to = google_storage_bucket.buckets["training-data"]
  id = "team17-mlops-neu-training-data"
}

import {
  to = google_storage_bucket.buckets["models"]
  id = "team17-mlops-neu-models"
}
