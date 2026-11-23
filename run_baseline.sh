#!/usr/bin/env bash
# Data pipeline, then the hourly baseline (repeated cross-validation and its report).
#
#   ./run_baseline.sh                  # both
#   ./run_baseline.sh --skip-data      # only the baseline, on the data already in data/processed
#
# Environment:
#   DATA_PYTHON   Python for Data_Pipeline   (default Data_Pipeline/.venv/bin/python)
#   MODEL_PYTHON  Python for Model_Pipeline  (default python3; needs Model_Pipeline/requirements.txt)
#   MLFLOW=1      log to MLflow instead of keeping tracking in memory
#
# The data pipeline processes data.participants in Data_Pipeline/config/pipeline.yaml (set it to the
# model's 60). Zips already in Data_Pipeline/data/raw are reused; others are downloaded from OSF.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_PYTHON="${DATA_PYTHON:-$ROOT/Data_Pipeline/.venv/bin/python}"
MODEL_PYTHON="${MODEL_PYTHON:-python3}"
# The script changes folder below, so make relative paths absolute now.
absolute() { if [[ "$1" == */* && "$1" != /* ]]; then echo "$PWD/$1"; else echo "$1"; fi; }
DATA_PYTHON="$(absolute "$DATA_PYTHON")"
MODEL_PYTHON="$(absolute "$MODEL_PYTHON")"
SKIP_DATA=0
[[ "${1:-}" == "--skip-data" ]] && SKIP_DATA=1

# macOS Python often lacks the system certificates (OSF downloads, DVC on GCS).
CERTS="$("$DATA_PYTHON" -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
[[ -n "$CERTS" ]] && export SSL_CERT_FILE="$CERTS" REQUESTS_CA_BUNDLE="$CERTS"
export MLFLOW_DISABLE_AGENT_HINT=1

step() { printf '\n==== %s ====\n' "$1"; }

# ---- 1. data pipeline --------------------------------------------------------------------------
cd "$ROOT/Data_Pipeline"
if [[ $SKIP_DATA -eq 0 ]]; then
  step "Data pipeline"
  N=$("$DATA_PYTHON" -c 'import yaml; p = yaml.safe_load(open("config/pipeline.yaml"))["data"]["participants"]; print(len(p or []))')
  echo "Participants in config/pipeline.yaml: $N"
  if [[ "$N" -lt 60 ]]; then
    read -r -p "Fewer than the model's 60. Continue anyway? [y/N] " ok
    [[ "$ok" == [yY] ]] || { echo "Stopped: set data.participants to the 60 first."; exit 1; }
  fi
  "$DATA_PYTHON" -m scripts.pipeline
fi

# ---- 2. the patient split (DVC) ----------------------------------------------------------------
step "Patient split"
if [[ ! -f data/processed/patient_split.json ]]; then
  echo "Pulling patient_split.json from DVC (needs: gcloud auth application-default login)"
  "$(dirname "$DATA_PYTHON")/dvc" pull data/processed/patient_split.json.dvc
fi
echo "Split: data/processed/patient_split.json"

# ---- 3. hourly baseline ------------------------------------------------------------------------
step "Hourly baseline"
cd "$ROOT/Model_Pipeline"
"$MODEL_PYTHON" -c 'import sklearn, mlflow, matplotlib' 2>/dev/null || {
  echo "MODEL_PYTHON ($MODEL_PYTHON) is missing packages: pip install -r Model_Pipeline/requirements.txt"; exit 1; }
FLAGS=(--baseline)
[[ "${MLFLOW:-0}" == "1" ]] || FLAGS+=(--no-mlflow)
"$MODEL_PYTHON" -m scripts.run_model_pipeline "${FLAGS[@]}"

step "Done"
echo "Report: $ROOT/Model_Pipeline/artifacts/baseline/report.md"
