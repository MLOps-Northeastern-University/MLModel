#!/usr/bin/env bash
# Sets up MedNOVA's deployment in a Google Cloud project, or checks one that's set up already:
# Terraform's state bucket, the Backend's first image, infra/ and infra/compute/, the database
# tables, then the status of each step that's done by hand. Each part is skipped when it's done
# already, so on the team's project it changes nothing.
#
#   infra/bootstrap.sh            set up, or finish setting up
#   infra/bootstrap.sh --check    change nothing: both terraform plans, /health, and what's left by hand
#
# Settings, as environment variables, each defaulting to the team's:
#   PROJECT_ID       where the Backend, MLflow and the buckets go      (team17-mlops-neu)
#   COMPUTE_PROJECT  where the training VMs run; PROJECT_ID works too  (finetuning-499422)
#   BACKEND_DIR      a checkout of the Backend repo, for its first image (../Backend, next to MLModel)
#   ALLOW_DESTROY=1  apply a plan that deletes or replaces something; otherwise it stops and shows it
# With MLModel's five alert settings set too (ALERT_EMAIL, SMTP_HOST, SMTP_PORT, SMTP_USER,
# SMTP_PASSWORD), the alert-emails and alert-smtp secrets are filled in when they're empty.
#
# Needs gcloud, terraform and python3; docker only to build the first image; gh to check GitHub.
set -euo pipefail

CHECK=0
case "${1:-}" in
  --check) CHECK=1 ;;
  "") ;;
  *) echo "usage: $0 [--check]" >&2; exit 2 ;;
esac

INFRA="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$INFRA")"
PROJECT_ID="${PROJECT_ID:-team17-mlops-neu}"
COMPUTE_PROJECT="${COMPUTE_PROJECT:-finetuning-499422}"
BACKEND_DIR="${BACKEND_DIR:-$(dirname "$REPO")/Backend}"
REGION=us-central1                       # variables.tf's region
STATE_BUCKET="$PROJECT_ID-tfstate"       # main.tf names the team's; init below swaps in this one
GITHUB_REPO=MLOps-Northeastern-University/MLModel
BACKEND_REPO=MLOps-Northeastern-University/Backend

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
LEFT=0   # what's still to do by hand, counted for the last line

say()  { printf '\n== %s\n' "$*"; }
ok()   { printf '  ok    %s\n' "$*"; }
todo() { printf '  TODO  %s\n' "$*"; LEFT=$((LEFT + 1)); }
note() { printf '        %s\n' "$*"; }
die()  { printf '\nStopped: %s\n' "$*" >&2; exit 1; }

secret_has_value() {   # secret_has_value <secret>: it exists in PROJECT_ID with an enabled version
  [ -n "$(gcloud secrets versions list "$1" --project "$PROJECT_ID" --filter=state=ENABLED --limit=1 \
    --format='value(name)' 2>/dev/null)" ]
}

# ---- 1. the tools --------------------------------------------------------------------------------

say "Tools"
missing=0
for tool in gcloud terraform python3; do
  if command -v "$tool" >/dev/null; then ok "$tool"; else todo "$tool is missing"; missing=1; fi
done
[ "$missing" = 0 ] || die "install what's missing: gcloud https://cloud.google.com/sdk/docs/install, terraform https://developer.hashicorp.com/terraform/install, python3 from python.org"
command -v gh >/dev/null && ok gh || note "gh is missing: GitHub's settings won't be checked (https://cli.github.com)"

# ---- 2. signed in, to the project -----------------------------------------------------------------

say "Google Cloud: $PROJECT_ID"
# Two logins: gcloud's own, and the one Terraform uses (application default credentials).
[ -n "$(gcloud auth list --filter=status:ACTIVE --format='value(account)')" ] || gcloud auth login
gcloud auth application-default print-access-token >/dev/null 2>&1 || gcloud auth application-default login
ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' | head -n 1)"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)' 2>/dev/null)" \
  || die "$ACCOUNT can't open project $PROJECT_ID: ask its owner for Owner on it"
ok "signed in as $ACCOUNT, project $PROJECT_ID ($PROJECT_NUMBER)"

# ---- 3. Terraform's state bucket ------------------------------------------------------------------

say "Terraform's state: gs://$STATE_BUCKET"
if gcloud storage buckets describe "gs://$STATE_BUCKET" --project "$PROJECT_ID" >/dev/null 2>&1; then
  ok "it exists"
elif [ "$CHECK" = 1 ]; then
  die "gs://$STATE_BUCKET doesn't exist: nothing is set up in $PROJECT_ID yet (run without --check)"
else
  # Terraform can't make the bucket it keeps its own state in. Versioning keeps every old state.
  gcloud storage buckets create "gs://$STATE_BUCKET" --project "$PROJECT_ID" --location "$REGION" \
    --uniform-bucket-level-access
  gcloud storage buckets update "gs://$STATE_BUCKET" --versioning
  # Terraform turns on every other API itself (apis.tf), through these two.
  gcloud services enable serviceusage.googleapis.com cloudresourcemanager.googleapis.com --project "$PROJECT_ID"
  ok "made it, with versioning"
fi

# The settings both folders read (variables.tf), for this project rather than the team's.
export TF_VAR_project_id="$PROJECT_ID"
export TF_VAR_compute_project="$COMPUTE_PROJECT"
export TF_VAR_training_vm_email="training-vm@$COMPUTE_PROJECT.iam.gserviceaccount.com"
export TF_VAR_ci_service_account="training@$PROJECT_ID.iam.gserviceaccount.com"

tf_init() {   # tf_init <folder>: its state in this project's bucket
  terraform -chdir="$1" init -input=false -reconfigure -backend-config="bucket=$STATE_BUCKET" \
    >"$WORK/init.log" 2>&1 || { cat "$WORK/init.log"; die "terraform init failed in $1"; }
}

# alert_emails_added (variables.tf) reads the alert-emails secret, which a new project doesn't have.
alert_emails_setting() {
  if secret_has_value alert-emails; then unset TF_VAR_alert_emails_added
  else export TF_VAR_alert_emails_added=false; fi
}

# The training VM's account is made in infra/compute/; until it exists, infra/'s grants to it can't be
# applied. NOT_FOUND means it doesn't exist; without access to COMPUTE_PROJECT it's taken to exist.
training_vm_exists() {
  local out
  out="$(gcloud iam service-accounts describe "$TF_VAR_training_vm_email" --project "$COMPUTE_PROJECT" 2>&1)" \
    && return 0
  ! grep -q NOT_FOUND <<<"$out"
}

changes() {   # changes <folder> <plan file>: one "<action> <address>" line per resource the plan changes
  terraform -chdir="$1" show -json "$2" | python3 -c '
import json, sys
for r in json.load(sys.stdin).get("resource_changes", []):
    actions = r["change"]["actions"]
    if actions in (["no-op"], ["read"]):
        continue
    print("replace" if "delete" in actions and "create" in actions else actions[0], r["address"])'
}

# plan_and_apply <folder> <label> [skip]: plans the folder and, unless --check, applies it. Resources
# whose address holds <skip> are left out, for a later run. A plan that deletes or replaces anything
# stops the script, unless ALLOW_DESTROY=1: mednova-db holds the patients' data.
plan_and_apply() {
  local folder=$1 label=$2 skip=${3:-} plan="$WORK/plan" list targets=()
  terraform -chdir="$folder" plan -input=false -lock-timeout=120s -out="$plan" >"$WORK/plan.log" 2>&1 \
    || { cat "$WORK/plan.log"; die "terraform plan failed in $label"; }
  list="$(changes "$folder" "$plan")"
  if [ -z "$list" ]; then ok "$label: No changes"; return 0; fi
  printf '%s\n' "$list" | sed 's/^/        /'
  if grep -qE '^(delete|replace) ' <<<"$list" && [ "${ALLOW_DESTROY:-}" != 1 ]; then
    die "$label's plan deletes or replaces the above. Read it (terraform -chdir=$folder plan), and run again with ALLOW_DESTROY=1 only if that's meant"
  fi
  if [ "$CHECK" = 1 ]; then
    local waiting=0
    [ -z "$skip" ] || waiting="$(grep -c "$skip" <<<"$list" || true)"
    if [ "$waiting" -gt 0 ]; then
      todo "$label: $(wc -l <<<"$list" | tr -d ' ') change(s) to apply; $waiting of them wait for the training VM's account (infra/compute/)"
    else
      todo "$label: $(wc -l <<<"$list" | tr -d ' ') change(s) to apply (run without --check)"
    fi
    return 0
  fi
  if [ -n "$skip" ] && grep -q "$skip" <<<"$list"; then
    note "left out for now: $(grep -c "$skip" <<<"$list") change(s) matching $skip"
    while read -r _ address; do targets+=("-target=$address"); done < <(grep -v "$skip" <<<"$list" || true)
    [ ${#targets[@]} -gt 0 ] || return 0
    terraform -chdir="$folder" plan -input=false -lock-timeout=120s -out="$plan" "${targets[@]}" \
      >"$WORK/plan.log" 2>&1 || { cat "$WORK/plan.log"; die "terraform plan failed in $label"; }
  fi
  terraform -chdir="$folder" apply -input=false -lock-timeout=120s "$plan"
  ok "$label: applied"
}

# ---- 4. the Backend's first image ------------------------------------------------------------------

# backend_run.tf makes the API and the jobs with this image, so it must exist before they do. After
# that the Backend's deploy workflow replaces it on every merge to main.
FIRST_TAG="$(sed -n 's#.*/ml/api:\([0-9a-f]*\)".*#\1#p' "$INFRA/backend_run.tf")"
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/ml/api:$FIRST_TAG"
say "The Backend's first image: ml/api:$FIRST_TAG"
tf_init "$INFRA"
alert_emails_setting
if gcloud artifacts docker images describe "$IMAGE" >/dev/null 2>&1; then
  ok "it's in the registry"
elif [ "$CHECK" = 1 ]; then
  todo "it isn't in the registry (run without --check)"
else
  [ -d "$BACKEND_DIR/.git" ] || die "no Backend checkout at $BACKEND_DIR: clone it there, or set BACKEND_DIR"
  command -v docker >/dev/null && docker info >/dev/null 2>&1 || die "docker isn't running: start Docker, then run again"
  git -C "$BACKEND_DIR" cat-file -e "$FIRST_TAG^{commit}" 2>/dev/null || git -C "$BACKEND_DIR" fetch -q origin
  # The registry first, alone: the rest of infra/ needs the image in it.
  terraform -chdir="$INFRA" apply -input=false -auto-approve -target=google_artifact_registry_repository.ml
  # The MLModel commit the Backend's deploys bake in now (its deploy.yml).
  MLMODEL_REF="$(sed -n 's/^ *MLMODEL_REF: *//p' "$BACKEND_DIR/.github/workflows/deploy.yml" | head -n 1)"
  mkdir "$WORK/backend"
  git -C "$BACKEND_DIR" archive "$FIRST_TAG" | tar -x -C "$WORK/backend"
  gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet
  # Cloud Run's CPUs are amd64, whatever this machine's are.
  docker build --platform linux/amd64 --build-arg MLMODEL_REF="${MLMODEL_REF:-main}" -t "$IMAGE" "$WORK/backend"
  docker push "$IMAGE"
  ok "built from Backend $FIRST_TAG and pushed"
fi

# ---- 5. Terraform: the team project, the training VMs' project, then the team project again -----

say "Terraform"
skip=""
training_vm_exists || skip=training_vm
plan_and_apply "$INFRA" "infra/ ($PROJECT_ID)" "$skip"

# The alert secrets, from MLModel's five settings, when they're set here and the secrets are empty.
refill=0
if [ "$CHECK" = 0 ] && [ -n "${ALERT_EMAIL:-}" ] && ! secret_has_value alert-emails; then
  printf '%s' "$ALERT_EMAIL" | gcloud secrets versions add alert-emails --project "$PROJECT_ID" --data-file=- >/dev/null
  ok "alert-emails filled in from ALERT_EMAIL"
  refill=1
fi
if [ "$CHECK" = 0 ] && [ -n "${SMTP_HOST:-}" ] && [ -n "${SMTP_PASSWORD:-}" ] && ! secret_has_value alert-smtp; then
  # {"to", "host", "port", "user", "password"}, which the training VM reads (airflow/vm/startup.sh).
  python3 -c 'import json, os; e = os.environ; print(json.dumps({"to": e["ALERT_EMAIL"], "host": e["SMTP_HOST"],
    "port": int(e.get("SMTP_PORT") or 587), "user": e.get("SMTP_USER", ""), "password": e["SMTP_PASSWORD"]}), end="")' \
    | gcloud secrets versions add alert-smtp --project "$PROJECT_ID" --data-file=- >/dev/null
  ok "alert-smtp filled in from the SMTP settings"
fi

if gcloud projects describe "$COMPUTE_PROJECT" >/dev/null 2>&1; then
  tf_init "$INFRA/compute"
  plan_and_apply "$INFRA/compute" "infra/compute/ ($COMPUTE_PROJECT)"
else
  todo "infra/compute/: $ACCOUNT can't open $COMPUTE_PROJECT. Its owner applies it, or set COMPUTE_PROJECT=$PROJECT_ID to train in this project"
fi

# Again for what the first pass left out, once it can go in: the grants to a training VM account
# made just now, and the alert emails once their secret has a value.
again=$refill
if [ -n "$skip" ] && training_vm_exists; then again=1; fi
if [ "$CHECK" = 0 ] && [ "$again" = 1 ]; then
  alert_emails_setting
  skip=""
  training_vm_exists || skip=training_vm
  plan_and_apply "$INFRA" "infra/ ($PROJECT_ID), again" "$skip"
fi

# ---- 6. the tables, and is it up -----------------------------------------------------------------

say "The Backend"
if [ "$CHECK" = 0 ]; then
  # alembic upgrade head: makes the tables, or does nothing when they're current.
  gcloud run jobs execute migrate --region "$REGION" --project "$PROJECT_ID" --wait >/dev/null
  ok "migrate ran: the tables are current"
fi
API_URL="$(terraform -chdir="$INFRA" output -raw BACKEND_API_URL 2>/dev/null || true)"
if [ -n "$API_URL" ] && health="$(curl -fsS --retry 3 --retry-delay 5 "$API_URL/health" 2>&1)"; then
  ok "$API_URL/health: $health"
else
  todo "the API doesn't answer at ${API_URL:-(no service yet)}/health. Its logs: Logs Explorer, resource.type=\"cloud_run_revision\" resource.labels.service_name=\"api\""
fi

# ---- 7. what's done by hand ------------------------------------------------------------------------

say "By hand"
# The listing kept in a variable: grep -q would stop reading early, and pipefail call that a failure.
if [ -n "$(gcloud storage ls "gs://$PROJECT_ID-training-data/harnet10/" 2>/dev/null | head -n 1)" ]; then
  ok "harnet10's weights are in training-data/harnet10/"
else
  todo "harnet10's weights into gs://$PROJECT_ID-training-data/harnet10/ (never in git, under their licence): gcloud storage cp -r <folder>/* gs://$PROJECT_ID-training-data/harnet10/"
fi
for secret in alert-emails alert-smtp; do
  if secret_has_value "$secret"; then ok "$secret has a value"
  else todo "$secret is empty: run this again with MLModel's five alert settings set, or add it by hand (variables.tf, secrets.tf)"; fi
done
if secret_has_value backend-firebase-web-api-key; then
  ok "backend-firebase-web-api-key has a value (firebase_web_api_key_added in variables.tf turns it on)"
else
  todo "backend-firebase-web-api-key is empty: the Firebase web app's key, then firebase_web_api_key_added = true (variables.tf). Without it new staff get no password email"
fi
note "Firebase is its own project: the Backend's accounts get their roles there by hand (Backend docs/DEPLOY.md, \"Firebase is in its own project\")"

if command -v gh >/dev/null; then
  mlflow_url="$(terraform -chdir="$INFRA" output -raw MLFLOW_TRACKING_URI 2>/dev/null || true)"
  provider="projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/providers"
  # check_github <repo> <NAME=value>...: the repo's variables, then the five alert secrets
  check_github() {
    local repo=$1 have secrets pair name value
    shift
    if ! have="$(gh variable list -R "$repo" --json name -q '.[].name' 2>/dev/null)"; then
      todo "$repo: gh can't read its settings (needs admin on the repo)"; return 0
    fi
    for pair in "$@"; do
      name=${pair%%=*} value=${pair#*=}
      if grep -qx "$name" <<<"$have"; then ok "$repo variable $name"
      else todo "$repo variable $name: gh variable set $name -R $repo --body '$value'"; fi
    done
    secrets="$(gh secret list -R "$repo" --json name -q '.[].name' 2>/dev/null || true)"
    for name in ALERT_EMAIL SMTP_HOST SMTP_PORT SMTP_USER SMTP_PASSWORD; do
      if grep -qx "$name" <<<"$secrets"; then ok "$repo secret $name"
      else todo "$repo secret $name: gh secret set $name -R $repo (it asks for the value)"; fi
    done
  }
  check_github "$GITHUB_REPO" "GCP_PROJECT_ID=$PROJECT_ID" "GCP_WIF_PROVIDER=$provider/mlmodel" \
    "GCP_SERVICE_ACCOUNT=training@$PROJECT_ID.iam.gserviceaccount.com" \
    "MLFLOW_TRACKING_URI=${mlflow_url:-<terraform output MLFLOW_TRACKING_URI>}" "GCP_COMPUTE_PROJECT=$COMPUTE_PROJECT"
  check_github "$BACKEND_REPO" "GCP_PROJECT_ID=$PROJECT_ID" "GCP_WIF_PROVIDER=$provider/backend" \
    "GCP_SERVICE_ACCOUNT=backend-deploy@$PROJECT_ID.iam.gserviceaccount.com"
fi

if [ "$LEFT" = 0 ]; then
  say "Done: nothing left to do"
else
  say "$LEFT thing(s) left, each marked TODO above"
fi
