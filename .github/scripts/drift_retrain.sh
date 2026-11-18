#!/usr/bin/env bash
# Turns the Backend's drift reports into GitHub issues, and an issue nobody closed into a retrain.
#
#   .github/scripts/drift_retrain.sh
#
# The Backend's nightly check (Backend app/scoring/drift.py) writes a report to
# gs://<team>-training-data/drift/<model>-<night>.json when the released model's P(OFF) has drifted
# from training's for 3 nights in a row. For each report not seen yet, this opens an issue
# "Drift: ... retrain in 24 h unless closed" (labelled drift-retrain), or comments on the one
# already open for that model. Closing the issue rejects the retrain. An issue still open after
# 24 hours starts model.yml on main, for the drifting model's candidate (cpu for logistic
# regression, gpu otherwise), releasing if it passes every gate; then it's labelled
# drift-retrained and closed. At most one drift retrain a week: a later one waits.
#
# Runs in CI (data-version.yml's sibling, drift.yml): GH_TOKEN is the workflow's own token, and
# gcloud is logged in as CI's training account. Environment: TEAM_PROJECT, GH_TOKEN.

set -euo pipefail

: "${TEAM_PROJECT:?set TEAM_PROJECT}" "${GH_TOKEN:?set GH_TOKEN}"
REPORTS="gs://$TEAM_PROJECT-training-data/drift"
WAIT_HOURS=24
COOLDOWN_DAYS=7

gh label create drift-retrain --color D93F0B --description "Drift: a retrain in 24 h unless closed" --force >/dev/null
gh label create drift-retrained --color 0E8A16 --description "Drift: the retrain was started" --force >/dev/null

# ---- 1. each new report: an issue (or a comment on the open one) -------------------------------

for report in $(gcloud storage ls "$REPORTS/*.json" 2>/dev/null || true); do
  marker="${report%.json}.issue"
  if gcloud storage cat "$marker" >/dev/null 2>&1; then continue; fi     # seen already
  gcloud storage cat "$report" > /tmp/report.json
  model_id="$(jq -r .model_id /tmp/report.json)"
  candidate="$(jq -r .candidate /tmp/report.json)"
  nights="$(jq -r '.nights[] | "| \(.night) | \(.psi) | \(.hours) | \(.patients) |"' /tmp/report.json)"

  open="$(gh issue list --label drift-retrain --state open --search "in:body \"model_id\":\"$model_id\"" \
    --json number --jq '.[0].number // empty')"
  if [ -n "$open" ]; then
    gh issue comment "$open" --body "Still drifting, as of $(jq -r .night /tmp/report.json):

| Night | PSI of P(OFF) | Hours | Patients |
|---|---|---|---|
$nights" >/dev/null
    printf '%s' "$open" | gcloud storage cp - "$marker"
    continue
  fi

  url="$(gh issue create --label drift-retrain \
    --title "Drift: $model_id's P(OFF) is unlike training's; retrain in 24 h unless closed" \
    --body "The released model ($model_id, $candidate) has said OFF unlike it did in training, by the Backend's nightly check ($(jq -r .rule /tmp/report.json)):

| Night | PSI of P(OFF) | Hours | Patients |
|---|---|---|---|
$nights

- **Do nothing** and, 24 hours after this issue was opened, a retrain of \`$candidate\` starts (\`model.yml\` on main). It's released only if it passes every gate and beats $model_id.
- **Close this issue** to reject the retrain.

<!-- drift: {\"model_id\":\"$model_id\",\"candidate\":\"$candidate\"} -->")"
  echo "opened $url for $report"
  printf '%s' "${url##*/}" | gcloud storage cp - "$marker"
done

# ---- 2. each issue still open after 24 hours: the retrain ------------------------------------

now="$(date -u +%s)"
last_retrain="$(gh issue list --label drift-retrained --state closed --json closedAt \
  --jq 'map(.closedAt | fromdateiso8601) | max // 0')"

gh issue list --label drift-retrain --state open --json number,createdAt,body \
  --jq '.[] | "\(.number) \(.createdAt | fromdateiso8601) \(.body | capture("<!-- drift: (?<j>[^>]*) -->").j | @base64)"' |
while read -r number created details; do
  if [ $(( now - created )) -lt $(( WAIT_HOURS * 3600 )) ]; then continue; fi
  if [ $(( now - last_retrain )) -lt $(( COOLDOWN_DAYS * 86400 )) ]; then
    echo "#$number waits: a drift retrain ran in the last $COOLDOWN_DAYS days"
    continue
  fi
  candidate="$(echo "$details" | base64 -d | jq -r .candidate)"
  model_id="$(echo "$details" | base64 -d | jq -r .model_id)"
  compute="gpu"
  case "$candidate" in window_*) compute="cpu" ;; esac          # logistic regression doesn't need a GPU
  gh workflow run model.yml --ref main -f candidates="$candidate" -f tune=false -f release=true \
    -f compute="$compute" -f reason="drift: $model_id's P(OFF) (issue #$number)"
  gh issue comment "$number" --body "Nobody closed this in $WAIT_HOURS hours: the retrain of \`$candidate\` (compute: $compute) has started. See Actions -> Model." >/dev/null
  gh issue edit "$number" --add-label drift-retrained >/dev/null
  gh issue close "$number" >/dev/null
  last_retrain="$now"
  echo "#$number: retrain started"
done
