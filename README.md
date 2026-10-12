# MLModel

The data and model pipelines for our Parkinson's monitoring system. It turns wrist-worn smartwatch movement data into symptom estimates (tremor severity, dyskinesia and slowness) for short time windows. The [Backend](https://github.com/MLOps-Northeastern-University/Backend) serves these estimates.

> **Status:** setting up. Tools aren't chosen yet. Sections marked _TBD_ get filled in once they are.

## What this repo does

- **Data pipeline:** loads the datasets, resamples sensors to a common rate, cuts fixed-length windows, joins clinician ratings to sensor windows, and validates the data
- **Model pipeline:** trains and evaluates models with subject-wise splits (no patient in both train and test), and tracks experiments
- **Promotion:** a new model replaces the current one only if it beats it on a fixed test set
- **Monitoring:** checks incoming data against the training data for drift

## Datasets

| Dataset | Use | Access |
|---|---|---|
| PADS | Learn what Parkinson's wrist movement looks like | Open download |
| MJFF Levodopa Response Study | Train and test the symptom severity model | Synapse. Each teammate must be approved individually. |
| Parkinson@Home | Check the model works in real homes | Repository terms apply |

The datasets themselves are **not** stored in this repo.

## Related repos

| Repo | Role |
|---|---|
| [AndroidApp](https://github.com/MLOps-Northeastern-University/AndroidApp) | Patient app (Flutter), collects watch data |
| [Backend](https://github.com/MLOps-Northeastern-University/Backend) | Serves this repo's model |
| [Careportal](https://github.com/MLOps-Northeastern-University/Careportal) | Clinician web portal |
| **MLModel** | This repo |

## Architecture

The options we considered for each technical choice and why we picked the one we did: [ARCHITECTURE.md](ARCHITECTURE.md).

## Getting started

_TBD once tools are chosen._ The goal is that one command runs each pipeline end to end.

## Running tests

_TBD._ CI will run on a small sample dataset, never on the gated datasets.

## Data rules

- **Never commit data, extracted features or model weights.** Git holds code only. Data and models go in access-controlled storage, tracked by a versioning tool (_TBD_).
- MJFF data stays with approved teammates under the Synapse terms. Don't share it or upload it to third-party services.
- Cite each dataset as its license requires.

## Checks

Every PR and every push to `dev` or `main` runs [.github/workflows/checks.yml](.github/workflows/checks.yml). Once branch protection requires them, a PR can't merge until they pass.

| Check | What it does |
|---|---|
| Secret scan | Fails if a key, token or password is committed (gitleaks) |
| Data file guard | Fails on data or model files (`.csv`, `.h5`, `.pt` and similar) or any file over 5 MB. Small synthetic test data may go in `tests/fixtures/`. |
| Markdown lint | Fails on broken Markdown. Rules are relaxed; see `.markdownlint-cli2.yaml`. |

Dependabot opens PRs to keep the workflow's actions up to date. Language checks get added here once the stack is chosen.

To run the Markdown check locally: `npx markdownlint-cli2 "**/*.md"`

## Contributing

`main` holds the stable version we demo and submit. `dev` is where work comes together. Nobody pushes directly to either.

1. Branch from `dev` as `<your-name>/<issue>-short-name`, e.g. `sam/7-pads-loader`.
2. Add a row to **Progress** below in the same PR.
3. Open a PR into `dev`. CI must pass and one teammate must approve.
4. Squash-merge into `dev` and delete the branch.

**Releases:** when `dev` is stable (end of a sprint, or before a graded submission), open a PR from `dev` into `main`. Merge it with **Create a merge commit**, not squash. Squashing here makes the two branches' histories diverge and causes false conflicts on every later release.

## Progress

Newest first. One row per merged PR.

| Date | Who | What changed | PR |
|---|---|---|---|
| 2026-10-12 | Sam Selvaraj | Added a tests line to the PR template | — |
| 2026-10-12 | Sam Selvaraj | Added `dev` branch flow: PRs go to `dev`, release PRs move `dev` into `main` | — |
| 2026-10-12 | Sam Selvaraj | Added CI checks (secret scan, data file guard, Markdown lint), .gitignore, PR template and Dependabot | — |
| 2026-10-12 | Sam Selvaraj | Added README and ARCHITECTURE.md | — |
