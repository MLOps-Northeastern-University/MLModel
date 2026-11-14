# MLModel

The data and model pipelines for our Parkinson's monitoring system. It turns wrist-worn smartwatch movement data into symptom estimates (tremor severity, dyskinesia and slowness) for short time windows. The [Backend](https://github.com/MLOps-Northeastern-University/Backend) serves these estimates. The data pipeline's first target is coarser: the hourly kinesia state (off, on or dyskinetic) from the patient's diary.

> **Status:** the data pipeline is complete, in [Data_Pipeline/](Data_Pipeline/README.md). The model pipeline is next.

## What this repo does

| Part | Status | What it does |
|---|---|---|
| **Data pipeline** | Done: [Data_Pipeline/README.md](Data_Pipeline/README.md) | Downloads the COPS dataset and builds the model's inputs: 10-second wrist windows at 30 Hz, every diary hour with its answers, every dose with its LEDD, and one row per patient. Checks them against the model's data contract, reports label balance per group, and versions the output with DVC. It runs as an Airflow DAG or as one command. |
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
| Data checks and anomalies | Our own checks with pandas against the model's data contract (`Data_Pipeline/scripts/checks.py`) |
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
| Model pipeline tests | flake8 and pytest on Python 3.11. The coverage floor is 0 while the steps are being built, and goes up to 85. |

A second workflow, [.github/workflows/model.yml](.github/workflows/model.yml), runs the model pipeline itself in the team's GCP project and emails the team when it fails or releases a model. For now it only runs by hand. How it works, its settings, and how to open the hosted MLflow: [Model_Pipeline/README.md, "Running in the cloud"](Model_Pipeline/README.md#running-in-the-cloud).

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
| 2026-11-14 | Tejaswini Sarikonda | Model pipeline (P3): the bias step (slices scored with Fairlearn, the bias gate with an alert, mitigation by reweighing as a new `_reweighted` candidate, `bias.json`, `bias_report.md` and plots) and the explain step (exact Shapley values over each model's input groups, the AUC drop when a group is left out, plots). Tested on synthetic data | (this PR) |
| 2026-11-14 | Dhruv Puri | Model pipeline (P1): evaluate and the validation gate, after train: a 95% interval for each pooled AUC from resampling patients, AUCs per fold and per patient, ROC plots and each candidate's `metrics.json`, with `gates_passed` set for select. Model pipeline reads the rebuilt data pipeline's output (`hours.csv`, `windows/manifest.json`) | (this PR) |
| 2026-11-14 | Dhruv Puri | Data pipeline rebuilt for the window model: 10-s 30 Hz windows from the more affected wrist with an orientation fix, every diary hour kept, dose events and a patients table, 18 modules down to 8. Run on the model team's 10 participants and pushed as a DVC data version | (this PR) |
| 2026-11-14 | Shaikh Basim | Model pipeline (P5): the GCP project as Terraform in `infra/` (buckets, Workload Identity Federation for CI, Artifact Registry, Cloud SQL, the hosted MLflow on Cloud Run), the run command through every step with stand-ins, the `model.yml` CI workflow, email alerts, and the shared DVC remote | (this PR) |
| 2026-11-13 | Ayush Varma | Model pipeline (P4): model selection, the release gate against production, the export bundle, and the registry with rollback. Placeholders for the phone runtime, the registry location and the Backend endpoint | (this PR) |
| 2026-11-11 | Sam Selvaraj | Model: the design decisions in MODEL.md. Model pipeline core: targets (kinesia score and tremor, now and next hour), the Dataset contract with a versioned window spec, the tracker contract with in-memory and MLflow trackers, and production logging. Data pipeline: dose events with LEDD from Jost et al. 2023. Fixed the Dependabot alerts | (this PR) |
| 2026-10-25 | Ayush Varma | Data pipeline run on 5 sample participants with `dvc repro` and Airflow, first DVC data version, results, bias findings, Gantt chart and screenshots in the docs, and this README updated | (this PR) |
| 2026-10-24 | Sam Selvaraj | Data pipeline: DVC, 7× faster processing, robustness fixes, CI, and tests (coverage 92%) | #5 |
| 2026-10-22 | Shaikh Basim | Data pipeline: fixed the run and the Airflow setup | #4 |
| 2026-10-21 | Tejaswini Sarikonda | Data pipeline: download, features, cleaning, validation, anomalies, bias, DAG, DVC stage, Docker and tests | #3 |
| 2026-10-12 | Sam Selvaraj | Repo setup: README, ARCHITECTURE.md, CI checks (secret scan, data file guard, Markdown lint) and the `dev` branch flow | — |
