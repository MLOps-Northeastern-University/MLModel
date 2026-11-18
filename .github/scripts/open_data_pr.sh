#!/usr/bin/env bash
# Opens the PR that records a new data version in git, for one data run.
#
#   .github/scripts/open_data_pr.sh <run id>
#
# A data run whose data changed leaves, in gs://<team>-training-data/runs/<run id>/data-version/
# (airflow/vm/startup.sh): the new dvc.lock, the bias report, dataset_meta.json (counts only, no
# participant IDs) and base-sha, the commit it ran. This opens a PR into main from that commit with
# the new dvc.lock and bias report: merge it to record the data (bump retrain.yaml in it to train
# on that data), close it to ignore the data. Then it writes pr.txt next to them, so a run gets
# one PR, however many times this runs (model.yml at the end of a run, data-version.yml hourly).
#
# Runs in CI, from the repo's checkout: GH_TOKEN (the workflow's own token, with contents and
# pull-requests write) opens the PR, and gcloud is logged in as CI's training account.
# Environment: TEAM_PROJECT, GH_TOKEN.

set -euo pipefail

RUN_ID="$1"
: "${TEAM_PROJECT:?set TEAM_PROJECT}" "${GH_TOKEN:?set GH_TOKEN}"
FOLDER="gs://$TEAM_PROJECT-training-data/runs/$RUN_ID/data-version"
BRANCH="data-version/$RUN_ID"

if gcloud storage cat "$FOLDER/pr.txt" > /dev/null 2>&1; then
  echo "$RUN_ID already has its PR: $(gcloud storage cat "$FOLDER/pr.txt")"
  exit 0
fi

# An earlier attempt that opened the PR but stopped before writing pr.txt: just record it.
existing="$(gh pr list --head "$BRANCH" --state all --json url --jq '.[0].url // empty')"
if [ -n "$existing" ]; then
  echo "$RUN_ID's PR is already open: $existing"
  printf '%s' "$existing" | gcloud storage cp - "$FOLDER/pr.txt"
  exit 0
fi

files="$(mktemp -d)"
gcloud storage cp "$FOLDER/*" "$files/" --quiet
base="$(cat "$files/base-sha")"

# The PR's description, from the counts.
summary="$(python3 - "$files/dataset_meta.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"{m['participants']} participants, {m['usable_hours']:,} usable hours of {m['hours']:,}, "
      f"{m['worn_windows']:,} worn windows, {m['doses']:,} doses (mode {m.get('mode')})")
PY
)"

git fetch -q origin "$base"
git switch -q -C "$BRANCH" "$base"
cp "$files/dvc.lock" Data_Pipeline/dvc.lock
cp "$files/bias_report.md" Data_Pipeline/data/reports/bias_report.md
git add Data_Pipeline/dvc.lock Data_Pipeline/data/reports/bias_report.md
git -c user.name="MedNOVA data pipeline" -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
  commit -q -m "Data version from run $RUN_ID: $summary"
# --force: the branch is this script's own; an earlier attempt may have pushed it without a PR.
git push -q --force origin "$BRANCH"

body="A data pipeline run made new data, now in the DVC remote: **$summary**.

- **Merge** to record this data version in git. The next training run uses it. To train on it straight away, bump \`request\` in \`Model_Pipeline/config/retrain.yaml\` in this PR before merging.
- **Close** to ignore it: nothing changes.

Run \`$RUN_ID\`, from commit $base. The changes are \`Data_Pipeline/dvc.lock\` (the new data's hashes) and the bias report."
url="$(gh pr create --base main --head "$BRANCH" --title "Data version: $summary" --body "$body")"
echo "opened $url"
printf '%s' "$url" | gcloud storage cp - "$FOLDER/pr.txt"
