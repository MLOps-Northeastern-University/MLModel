# MLModel

The data and model pipelines for our Parkinson's monitoring system. It turns wrist-worn smartwatch movement data into symptom estimates (tremor severity, dyskinesia and slowness) for short time windows. The [Backend](https://github.com/MLOps-Northeastern-University/Backend) serves these estimates. The data pipeline's first target is coarser: the hourly kinesia state (off, on or dyskinetic) from the patient's diary.

> **Status:** the data pipeline is complete, in [Data_Pipeline/](Data_Pipeline/README.md). The model pipeline is next.

## What this repo does

| Part | Status | What it does |
|---|---|---|
| **Data pipeline** | Done: [Data_Pipeline/README.md](Data_Pipeline/README.md) | Downloads the COPS dataset, turns wrist accelerometry and the symptom diary into one row of features and a label for each participant-hour, then cleans, validates, checks for anomalies and bias, and versions the output with DVC. It runs as an Airflow DAG or as one command. |
| **Model pipeline** | Next | Trains and evaluates models with subject-wise splits (no patient in both train and test), and tracks experiments |
| **Promotion** | Planned | A new model replaces the current one only if it beats it on a fixed test set |
| **Monitoring** | Planned | Checks incoming data against the training data for drift |

## Dataset

| Dataset | Use | Access |
|---|---|---|
| COPS ([osf.io/5xvwn](https://osf.io/5xvwn/)) | People with Parkinson's at home: both wrists at 100 Hz, an hourly symptom diary (the label: off, on, or dyskinetic), MDS-UPDRS III exams, and demographics. 66 participants, 47.9 GB. | Public download from OSF. The pipeline downloads it. |

The dataset itself is **not** stored in this repo. DVC versions the pipeline's outputs.

## Related repos

| Repo | Role |
|---|---|
| [AndroidApp](https://github.com/MLOps-Northeastern-University/AndroidApp) | Patient app (native Android, Java), collects watch data |
| [Backend](https://github.com/MLOps-Northeastern-University/Backend) | Serves this repo's model |
| [Careportal](https://github.com/MLOps-Northeastern-University/Careportal) | Clinician web portal |
| **MLModel** | This repo |

## Architecture

The options we considered for each technical choice and why we picked the one we did: [ARCHITECTURE.md](ARCHITECTURE.md).

What the model has to produce for the dashboard, the data and inputs it can use, and the decisions made about it: [MODEL.md](MODEL.md).

## Tools

| Job | Tool |
|---|---|
| Orchestration | Airflow 2.10 in Docker (`Data_Pipeline/docker-compose.yaml`) |
| Data versioning | DVC (`Data_Pipeline/dvc.yaml`, `dvc.lock`) |
| Schema, statistics and anomalies | Our own checks with pandas (`Data_Pipeline/scripts/validate.py`, `anomalies.py`) |
| Bias slicing | Fairlearn |
| Tests and style | pytest (coverage floor 85%), flake8 |

## Getting started

Only Docker is needed. From `Data_Pipeline/`:

```bash
docker compose run --rm tests       # the tests (the first run builds the image)
docker compose run --rm pipeline    # the pipeline on 5 participants, 1.08 GB download
```

On Linux, run `mkdir -p logs/airflow && echo "AIRFLOW_UID=$(id -u)" > .env` once first, so the containers can write to the folder. To run it in Airflow, or without Docker on Python 3.11 or 3.12, see the [quick start in Data_Pipeline/README.md](Data_Pipeline/README.md#quick-start-only-docker-needed).

## Running tests

From `Data_Pipeline/`: `docker compose run --rm tests`, or `pytest -v` with Python 3.11 or 3.12 (which skips the Airflow DAG tests). The tests use small synthetic files, so they download nothing.

## Data rules

- **Never commit data, extracted features or model weights.** Git holds code only. DVC versions the data. Only the summary reports with no participant IDs are in Git.
- Cite the COPS dataset as its OSF page requires.

## Checks

Every PR and every push to `dev` or `main` runs [.github/workflows/checks.yml](.github/workflows/checks.yml). Once branch protection requires them, a PR can't merge until they pass.

| Check | What it does |
|---|---|
| Secret scan | Fails if a key, token or password is committed (gitleaks) |
| Data file guard | Fails on data or model files (`.csv`, `.h5`, `.pt` and similar) or any file over 5 MB. Small synthetic test data may go in `tests/fixtures/`. |
| Markdown lint | Fails on broken Markdown. Rules are relaxed; see `.markdownlint-cli2.yaml`. |
| Data pipeline tests | flake8, pytest with an 85% coverage floor, and `dvc dag`, on Python 3.11 |

Dependabot opens PRs to keep the workflow's actions up to date.

To run the Markdown check locally: `npx markdownlint-cli2 "**/*.md"`

## Contributing

`main` holds the stable version we demo and submit. `dev` is where work comes together. Nobody pushes directly to either.

1. Branch from `dev` as `<your-name>/<issue>-short-name`, e.g. `sam/7-pads-loader`.
2. Add a row to **Progress** below in the same PR.
3. Open a PR into `dev`. CI must pass and one teammate must approve.
4. Squash-merge into `dev` and delete the branch.

**Releases:** when `dev` is stable (end of a sprint, or before a graded submission), open a PR from `dev` into `main`. Merge it with **Create a merge commit**, not squash. Squashing here makes the two branches' histories diverge and causes false conflicts on every later release.

## Progress

Newest first. One row per milestone, not per small change.

| Date | Who | What changed | PR |
|---|---|---|---|
| 2026-10-25 | Ayush Varma | Data pipeline run on 5 sample participants with `dvc repro` and Airflow, first DVC data version, results, bias findings, Gantt chart and screenshots in the docs, and this README updated | (this PR) |
| 2026-10-24 | Sam Selvaraj | Data pipeline: DVC, 7× faster processing, robustness fixes, CI, and tests (coverage 92%) | #5 |
| 2026-10-22 | Shaikh Basim | Data pipeline: fixed the run and the Airflow setup | #4 |
| 2026-10-21 | Tejaswini Sarikonda | Data pipeline: download, features, cleaning, validation, anomalies, bias, DAG, DVC stage, Docker and tests | #3 |
| 2026-10-12 | Sam Selvaraj | Repo setup: README, ARCHITECTURE.md, CI checks (secret scan, data file guard, Markdown lint) and the `dev` branch flow | — |
