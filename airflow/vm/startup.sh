#!/usr/bin/env bash
# The training VM's start-up script: one training run, then the VM shuts down.
#
# CI makes the VM (.github/workflows/model.yml) with this file as its startup-script and the run's
# settings as metadata, and deletes it once the run's result is in GCS. GCE runs the script as root
# at every boot; the NVIDIA driver's installer reboots once, so every step checks whether it's done.
#
#   1. machine   Docker, and for a GPU run the NVIDIA driver and container toolkit
#   2. code      the repo at the run's commit (public on GitHub), and the run's images
#   3. files     the data (dvc pull), harnet10's weights and what runs keep, from GCS
#   4. Airflow   started with each teammate's login; its API off, so only the login page answers
#   5. the run   the DAG triggered with the run's settings, a Cloudflare tunnel to the UI, the
#                link emailed to the team (never printed in a public log)
#   6. after     what runs keep, the logs and the result back to GCS, then the VM deletes itself
#
# The run's metadata (set by CI):
#   mednova-run-id        a name for this run, unique (CI: gh-<run id>-<attempt>)
#   mednova-git-sha       the commit to run
#   mednova-dag           model_pipeline or cops_data_pipeline
#   mednova-dag-conf      the DAG's trigger config, JSON
#   mednova-airflow-image, mednova-data-image, mednova-model-image   the images CI pushed
#   mednova-gpu           1 for a GPU run
#   mednova-mlflow-uri    the team MLflow
#   mednova-team-project  the team project, whose buckets, secrets and images it uses
#
# What it reads and writes in GCS (the team project's buckets, private):
#   gs://<team>-training-data/cops-dvc/        the DVC data (dvc pull)
#   gs://<team>-training-data/harnet10/        harnet10's weights, uploaded once by hand
#   gs://<team>-training-data/state/           what runs keep: the feature caches, tuned settings,
#                                              the tuning studies, the test-scored markers
#   gs://<team>-training-data/runs/<run id>/   this run's result.json (CI waits for it) and logs;
#                                              for a data run that changed the data, data-version/
#                                              (dvc.lock and its reports, for CI's PR)
#
# Its own log: the VM's serial console, and runs/<run id>/startup.log.

# -E: an error inside a function also reaches the ERR trap below, which records the run as failed.
set -Eeuo pipefail

REPO_URL="https://github.com/MLOps-Northeastern-University/MLModel.git"
WORK="/srv/mednova"
REPO="$WORK/MLModel"
HARNET="$WORK/harnet10"
# The longest a run may take before it's stopped and recorded as failed (CI also gives the VM a
# maximum run time, in case this script itself hangs).
MAX_RUN_HOURS=12

log() { echo "[mednova $(date -u +%H:%M:%S)] $*"; }

metadata() {   # metadata <key>: one of the VM's metadata values, or empty
  curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" || true
}

end_vm() {     # delete this VM (it may delete only itself: infra/compute/), or at least shut it down
  local zone
  zone="$(curl -sf -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/zone")"
  gcloud compute instances delete "$(hostname)" --zone "${zone##*/}" --quiet || shutdown -h now
}

secret() {     # secret <name>: the latest version of a Secret Manager secret, or empty
  gcloud secrets versions access latest --secret "$1" --project "$TEAM_PROJECT" 2>/dev/null || true
}

# ---- the run's settings ----------------------------------------------------------------------

RUN_ID="$(metadata mednova-run-id)"
SHA="$(metadata mednova-git-sha)"
DAG="$(metadata mednova-dag)"
DAG_CONF="$(metadata mednova-dag-conf)"
AIRFLOW_IMAGE="$(metadata mednova-airflow-image)"
DATA_IMAGE="$(metadata mednova-data-image)"
MODEL_IMAGE="$(metadata mednova-model-image)"
GPU="$(metadata mednova-gpu)"
MLFLOW_URI="$(metadata mednova-mlflow-uri)"
[ -n "$DAG_CONF" ] || DAG_CONF="{}"
# The team project: its buckets, secrets and images. The VM itself may run in another project
# (infra/compute/, for its GPU quota), so this comes from the metadata, not the VM's own project.
TEAM_PROJECT="$(metadata mednova-team-project)"
BUCKET="gs://${TEAM_PROJECT}-training-data"
RUN_DIR="$BUCKET/runs/$RUN_ID"

# Everything below goes to the serial console and to a file copied to GCS at the end.
mkdir -p "$WORK"
exec > >(tee -a "$WORK/startup.log") 2>&1

for name in RUN_ID SHA DAG AIRFLOW_IMAGE DATA_IMAGE MODEL_IMAGE TEAM_PROJECT; do
  if [ -z "${!name}" ]; then log "missing metadata for $name: CI sets it"; end_vm; exit 1; fi
done

# A reboot after the run finished (the VM is waiting to be deleted): nothing more to do.
if gcloud storage ls "$RUN_DIR/result.json" >/dev/null 2>&1; then
  log "run $RUN_ID already has a result; deleting this VM"
  end_vm
  exit 0
fi

# Whatever happens next, the run gets a result and the VM shuts down, so a broken run never sits
# there costing money.
finish() {
  trap - ERR                   # a failure in here mustn't call finish again
  local state="$1" detail="${2:-}"
  log "run $RUN_ID: $state $detail"
  keep_state || true
  if [ -d "$REPO/airflow/logs" ]; then
    gcloud storage rsync -r "$REPO/airflow/logs" "$RUN_DIR/airflow-logs" || true
  fi
  local released=""
  if [ -f "$REPO/Model_Pipeline/artifacts/run.json" ]; then
    released="$(python3 -c "import json; print(json.load(open('$REPO/Model_Pipeline/artifacts/run.json')).get('released_version') or '')")"
  fi
  python3 - "$state" "$detail" "$released" > "$WORK/result.json" <<'PY'
import datetime, json, sys
state, detail, released = sys.argv[1:4]
print(json.dumps({"state": state, "detail": detail, "released_version": released or None,
                  "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}))
PY
  gcloud storage cp "$WORK/startup.log" "$RUN_DIR/startup.log" || true
  gcloud storage cp "$WORK/result.json" "$RUN_DIR/result.json" || true
  end_vm
}
trap 'finish failed "the start-up script stopped at line $LINENO"' ERR

# ---- 1. the machine --------------------------------------------------------------------------

install_docker() {
  if command -v docker >/dev/null && docker compose version >/dev/null 2>&1; then return; fi
  log "installing Docker"
  apt-get update -q
  apt-get install -y -q docker.io docker-compose-v2
  systemctl enable --now docker
}

install_nvidia() {
  if nvidia-smi >/dev/null 2>&1 && command -v nvidia-ctk >/dev/null; then return; fi
  if ! nvidia-smi >/dev/null 2>&1; then
    log "installing the NVIDIA driver (the installer reboots the VM; this script runs again after)"
    curl -sfL https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz \
      -o "$WORK/cuda_installer.pyz"
    python3 "$WORK/cuda_installer.pyz" install_driver
    reboot     # it usually reboots by itself; the driver loads at the next boot either way
    exit 0
  fi
  log "installing NVIDIA's container toolkit, so containers can use the GPU"
  curl -sfL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -sfL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -q
  apt-get install -y -q nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
}

install_docker
if [ "$GPU" = "1" ]; then install_nvidia; fi

# ---- 2. the code and the images --------------------------------------------------------------

log "the repo at $SHA"
if [ ! -d "$REPO/.git" ]; then git clone -q "$REPO_URL" "$REPO"; fi
git -C "$REPO" fetch -q origin "$SHA"
git -C "$REPO" checkout -q --force "$SHA"

log "pulling the images"
gcloud auth configure-docker "${AIRFLOW_IMAGE%%/*}" --quiet
for image in "$AIRFLOW_IMAGE" "$DATA_IMAGE" "$MODEL_IMAGE"; do docker pull -q "$image"; done

# ---- 3. the files ----------------------------------------------------------------------------

ARTIFACTS="$REPO/Model_Pipeline/artifacts"
STATE="$BUCKET/state"

# What one run leaves for the next (each is small, or worth keeping: harnet10's features take
# hours to compute on the full data).
KEPT=("cache:$ARTIFACTS/cache" "tuning:$ARTIFACTS/tuning" "test_scored:$ARTIFACTS/test_scored")

restore_state() {
  mkdir -p "$ARTIFACTS"
  for entry in "${KEPT[@]}"; do
    local name="${entry%%:*}" folder="${entry#*:}"
    mkdir -p "$folder"
    gcloud storage rsync -r "$STATE/$name" "$folder" 2>/dev/null || true
  done
  gcloud storage cp "$STATE/tuned_params.json" "$ARTIFACTS/tuned_params.json" 2>/dev/null || true
}

keep_state() {
  for entry in "${KEPT[@]}"; do
    local name="${entry%%:*}" folder="${entry#*:}"
    [ -d "$folder" ] && gcloud storage rsync -r "$folder" "$STATE/$name"
  done
  [ -f "$ARTIFACTS/tuned_params.json" ] && gcloud storage cp "$ARTIFACTS/tuned_params.json" "$STATE/tuned_params.json"
  return 0
}

if [ "$DAG" = "model_pipeline" ]; then
  log "the data this commit names, from DVC"
  # In the data pipeline's image (it has DVC and git), logged in as this VM's account.
  docker run --rm -v "$REPO:/app" -w /app/Data_Pipeline --entrypoint dvc "$DATA_IMAGE" pull -q
  log "harnet10's weights"
  mkdir -p "$HARNET"
  gcloud storage rsync -r "$BUCKET/harnet10" "$HARNET"
  log "what earlier runs kept"
  restore_state
fi

# ---- 4. Airflow ------------------------------------------------------------------------------

AIRFLOW="$REPO/airflow"
USERS_JSON="$(secret airflow-users)"
if [ -z "$USERS_JSON" ]; then finish failed "no airflow-users secret: terraform apply makes it"; exit 1; fi

# The alert settings: alert-smtp is {"to", "host", "port", "user", "password"}.
SMTP_JSON="$(secret alert-smtp)"
smtp() {       # smtp <field>: one field of alert-smtp, or empty
  if [ -n "$SMTP_JSON" ]; then
    python3 -c "import json,sys; print(json.loads(sys.argv[1]).get(sys.argv[2], ''))" "$SMTP_JSON" "$1"
  fi
}

FIRST_USER="$(python3 -c "import json,sys; print(next(iter(json.loads(sys.argv[1]))))" "$USERS_JSON")"
FIRST_PASSWORD="$(python3 -c "import json,sys; u=json.loads(sys.argv[1]); print(u[next(iter(u))])" "$USERS_JSON")"

cat > "$AIRFLOW/.env" <<EOF
HOST_REPO=$REPO
HARNET_DIR=$HARNET
AIRFLOW_IMAGE=$AIRFLOW_IMAGE
DATA_IMAGE=$DATA_IMAGE
MODEL_IMAGE=$MODEL_IMAGE
MODEL_GPU=${GPU:-0}
GIT_COMMIT=$SHA
MLFLOW_TRACKING_URI=$MLFLOW_URI
MODEL_MLFLOW_GOOGLE_AUTH=1
MODEL_REGISTRY=gs://${TEAM_PROJECT}-models
AIRFLOW_UID=50000
DOCKER_GID=$(stat -c %g /var/run/docker.sock)
AIRFLOW_ADMIN_USER=$FIRST_USER
AIRFLOW_ADMIN_PASSWORD=$FIRST_PASSWORD
AIRFLOW_API_AUTH=airflow.api.auth.backend.deny_all
ALERT_EMAIL=$(smtp to)
SMTP_HOST=$(smtp host)
SMTP_PORT=$(smtp port)
SMTP_USER=$(smtp user)
SMTP_PASSWORD=$(smtp password)
SLACK_WEBHOOK_URL=$(secret alert-slack-webhook)
EOF
chmod 600 "$AIRFLOW/.env"
mkdir -p "$AIRFLOW/logs" && chown -R 50000:0 "$AIRFLOW/logs"

compose() { docker compose -p mednova -f "$AIRFLOW/docker-compose.yaml" --project-directory "$AIRFLOW" "$@"; }
airflow_cli() { compose exec -T airflow-scheduler airflow "$@"; }

log "starting Airflow"
compose up --no-build airflow-init
compose up --no-build -d airflow-webserver airflow-scheduler
for _ in $(seq 1 60); do curl -sf http://localhost:8080/health >/dev/null && break; sleep 5; done

log "the team's logins"
python3 -c "import json,sys; [print(u, p) for u, p in json.loads(sys.argv[1]).items()]" "$USERS_JSON" |
while read -r user password; do
  [ "$user" = "$FIRST_USER" ] && continue      # made by airflow-init
  airflow_cli users create --username "$user" --password "$password" --role Admin \
    --firstname "$user" --lastname MedNOVA --email "$user@mednova.invalid" >/dev/null 2>&1 || true
done

# ---- 5. the run ------------------------------------------------------------------------------

log "the Airflow UI through a Cloudflare tunnel (outbound only: no port is open on this VM)"
docker rm -f mednova-tunnel >/dev/null 2>&1 || true
docker run -d --name mednova-tunnel --network host cloudflare/cloudflared:2024.12.2 \
  tunnel --no-autoupdate --url http://localhost:8080 >/dev/null
LINK=""
for _ in $(seq 1 30); do
  LINK="$(docker logs mednova-tunnel 2>&1 | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' | head -1 || true)"
  [ -n "$LINK" ] && break
  sleep 2
done

# The link goes to the team only, through the alert channels (it's the way to the login page):
# airflow/dags/alerts.py, run in the Airflow container, which has the alert settings from .env.
notify() {
  compose exec -T airflow-scheduler python -c '
import sys
sys.path.insert(0, "/opt/airflow/dags")
from alerts import send_email, send_slack
subject, body = sys.argv[1:3]
send_email(subject, body)
send_slack(f"*{subject}*
{body}")' "$1" "$2" || true
}

log "waiting for Airflow to load $DAG"
for _ in $(seq 1 60); do airflow_cli dags list -o plain 2>/dev/null | grep -qw "$DAG" && break; sleep 5; done

log "triggering $DAG"
airflow_cli dags unpause "$DAG" >/dev/null
airflow_cli dags trigger "$DAG" --run-id "$RUN_ID" --conf "$DAG_CONF" >/dev/null

notify "[MedNOVA] $DAG is running ($RUN_ID)" "Commit: $SHA
Settings: $DAG_CONF

Airflow (log in with your MedNOVA Airflow account): ${LINK:-none: the tunnel did not start, see the VM log}
MLflow: $MLFLOW_URI

The link works while the run lasts; the VM is deleted when it ends."

log "waiting for $DAG to finish"
STATE=""
DEADLINE=$(( $(date +%s) + MAX_RUN_HOURS * 3600 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  STATE="$(airflow_cli dags list-runs -d "$DAG" -o json 2>/dev/null |
    python3 -c "import json,sys; print(next((r['state'] for r in json.load(sys.stdin) if r['run_id'] == '$RUN_ID'), ''))" || true)"
  case "$STATE" in success|failed) break ;; esac
  sleep 60
done
DETAIL=""
if [ "$STATE" != "success" ] && [ "$STATE" != "failed" ]; then
  STATE="failed"
  DETAIL="took longer than $MAX_RUN_HOURS hours"
fi

# ---- 6. after --------------------------------------------------------------------------------

# A data run that changed the data: its dvc_version task updated dvc.lock (and DVC pushed the data).
# What git needs to record it goes to GCS, and CI opens the PR (.github/scripts/open_data_pr.sh),
# so no GitHub credential is ever on this VM. Nothing if the data came out the same.
publish_data_version() {
  local lock="Data_Pipeline/dvc.lock" out="$WORK/data-version"
  if git -C "$REPO" diff --quiet -- "$lock"; then
    log "the data is the same as the commit's: no new data version"
    return
  fi
  log "a new data version: its dvc.lock goes to GCS for CI's PR"
  mkdir -p "$out"
  cp "$REPO/$lock" "$out/dvc.lock"
  cp "$REPO/Data_Pipeline/data/reports/bias_report.md" "$out/bias_report.md"
  # Counts only, no participant IDs: the PR's description.
  cp "$REPO/Data_Pipeline/data/processed/dataset_meta.json" "$out/dataset_meta.json"
  printf '%s' "$SHA" > "$out/base-sha"
  gcloud storage cp "$out"/* "$RUN_DIR/data-version/"
}

trap - ERR
if [ "$DAG" = "cops_data_pipeline" ] && [ "$STATE" = "success" ]; then
  publish_data_version || STATE="failed" DETAIL="couldn't save the new data version"
fi
finish "$STATE" "$DETAIL"
