# The buckets were made by hand with gcloud, so Terraform takes them over instead of
# creating them again. Only works if these names exist exactly (check with
# `gcloud storage ls --project=team17-mlops-neu`). Once they're in the state this
# file does nothing, so it can be deleted after the first apply.

# Import blocks (Terraform 1.5 and later) take effect at plan and apply: the plan lists each
# bucket as "will be imported", not created. If a name didn't exist, the plan would stop with
# an error instead of making a new, empty bucket.
import {
  to = google_storage_bucket.buckets["training-data"]
  # id is the bucket's name in GCP; to is the storage.tf entry it becomes.
  id = "team17-mlops-neu-training-data"
}

import {
  to = google_storage_bucket.buckets["models"]
  id = "team17-mlops-neu-models"
}
