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
| Data checks and anomalies | Our own checks with pandas against the model's data contract (`Data_Pipeline/data_pipeline/checks.py`) |
| Bias slicing | Fairlearn |
| Tests and style | pytest (coverage floor 85%), flake8 |

## Getting started

Only Docker is needed. From `Data_Pipeline/`:

```bash
docker compose run --rm tests       # the tests (the first run builds the image)
docker compose run --rm pipeline    # the pipeline on 5 participants, 1.08 GB download
```

To run it without Docker, on Python 3.12, see the [quick start in Data_Pipeline/README.md](Data_Pipeline/README.md#quick-start-only-docker-needed). Both pipelines also run as DAGs on one Airflow, in [airflow/](airflow/README.md).

## Running tests

From `Data_Pipeline/` or `Model_Pipeline/`: `docker compose run --rm tests`, or `pytest -v` with Python 3.12. The DAGs' tests run in the Airflow image: `docker compose run --rm tests` in `airflow/`. The tests use small synthetic files, so they download nothing.

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
| Data pipeline tests | flake8, pytest with an 85% coverage floor, and `dvc dag`, on Python 3.12 |
| Model pipeline tests | flake8 and pytest on Python 3.12. The coverage floor is 0 while the steps are being built, and goes up to 85. |
| Airflow DAGs | both DAGs load, connect in order and run each step in its pipeline's image, in the Airflow image |

A second workflow, [.github/workflows/model.yml](.github/workflows/model.yml), trains the model on a VM made for the run: a T4 or L4 GPU machine, or a CPU one, in the first zone GCP has free. It runs when a merge into `main` bumps `Model_Pipeline/config/retrain.yaml`'s `request`, or by hand (Actions → Model → Run workflow), runs the model DAG in Airflow on the VM, logs to the hosted MLflow, and emails the team the result. How it works, its settings, and how it maps to the assignment's CI/CD section: [Model_Pipeline/README.md, "Running in the cloud"](Model_Pipeline/README.md#running-in-the-cloud).

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
| 2026-11-25 | Shaikh Basim | Monitoring for the deployed Backend (`monitoring.tf`, applied): an uptime check on the API's `/health` every 15 minutes, and alerts that email the team when the API is down, answers 5 or more 5xx in 5 minutes, a job fails (after its retry), or `score-hours` hasn't succeeded for 3 hours. Only failures email, never a success or a cleared alert. The addresses are in Secret Manager (`alert-emails`), not the repo. A dashboard, "MedNOVA Backend": requests, latency, 5xx, instances and job runs | (this PR) |
| 2026-11-24 | Shaikh Basim | The Backend's jobs on Cloud Run and their schedules (`backend_run.tf`, applied): `migrate`, which the deploy runs before each new version, then `score-hours` hourly and `model-metrics`, `check-drift` and `end-and-purge` nightly, all from the API's image as `backend-scoring`. Cloud Scheduler starts them as `backend-scheduler`, which may only start them. `migrate` has made the tables. `score-hours` and `check-drift` stay paused (`scoring_paused`) until harnet10's weights are in `training-data/harnet10/` | (this PR) |
| 2026-11-23 | Shaikh Basim | The Backend's API on Cloud Run (`backend_run.tf`, applied): the `api` service as `backend-api`, with the database connection, its two secrets, public access (the API checks Firebase sign-ins itself), one instance at most for `mednova-db`'s connections, and the image left to the Backend's deploy workflow. `careportal_origins` and `firebase_project` wait on the portal's address and the Firebase project. Comments through the pipelines, CI and `infra/` on why things are as they are | (this PR) |
| 2026-11-23 | Shaikh Basim | The Backend's deploys sign in to Google Cloud without a key (`ci_access.tf`, applied): a `backend` provider in the `github` pool that trusts only the Backend repository, and the `backend-deploy` account its deploy workflow runs as, which may push to `ml`, update Cloud Run, and run it as `backend-api` or `backend-scoring` only. Tested from the Backend repo | (this PR) |
| 2026-11-22 | Shaikh Basim | The Backend's deploy, first part, as Terraform in `infra/` (applied): the Cloud Scheduler, Logging and Monitoring APIs; the Backend's database and user on `mednova-db`, with `DATABASE_URL` and `SECRET_KEY` in Secret Manager (`backend.tf`); and the two accounts it runs as, `backend-api` for the public API and `backend-scoring` for the scoring jobs, each with only what it uses (the jobs reach `harnet10/` and `drift/` in the training-data bucket, not the licensed data) | (this PR) |
| 2026-11-18 | Sam Selvaraj | Training as a production loop: one Airflow for both pipelines (`airflow/`), each step in its pipeline's image, on a training VM CI makes per run (a T4, else an L4, or an e2 for CPU, in the first zone GCP has free) and that deletes itself; `retrain.yaml` decides what trains on a merge; `--candidates` and `--tune`; every model step runnable on its own; CI opens a PR for each new data version, and turns the Backend's drift reports into issues that retrain if left open 24 h. Terraform for the VMs (`infra/compute/`) and their access. The packages renamed `data_pipeline` and `model_pipeline`, Python 3.12, and sample mode no longer downloads every participant | (this PR) |
| 2026-11-15 | Sam Selvaraj | First full run on real data: the 66-patient split (10 test, 6 demo, 50 development), the data pipeline through Airflow on 60 participants with the data and split under DVC, then tuning and the whole model pipeline on a GCP GPU machine, logged to the hosted MLflow. Model pipeline (P2): each patient compared with their own past week (within-patient OFF from chance to about 0.64), early stopping by OFF, charts per run, and leakage and bias gates that judge what they should. Docs updated with the results and what was tested | #22 |
| 2026-11-15 | Ayush Varma | Model pipeline (P4): the registry in the models bucket (`MODEL_REGISTRY` in CI), a `status` and `rollback` command, and the release gate skips a metric neither model has | (this PR) |
| 2026-11-15 | Ayush Varma | Model pipeline (P4): export for the server (the saved model folder, manifest, metrics, stats and bias report; harnet10 not bundled), optional device export off by default, and select no longer needs `exportable` | (this PR) |
| 2026-11-14 | Tejaswini Sarikonda | Model pipeline (P3): the bias step (slices scored with Fairlearn, the bias gate with an alert, mitigation by reweighing as a new `_reweighted` candidate, `bias.json`, `bias_report.md` and plots) and the explain step (exact Shapley values over each model's input groups, the AUC drop when a group is left out, plots). Tested on synthetic data | (this PR) |
| 2026-11-14 | Dhruv Puri | Model pipeline (P1): evaluate and the validation gate, after train: a 95% interval for each pooled AUC from resampling patients, AUCs per fold and per patient, ROC plots and each candidate's `metrics.json`, with `gates_passed` set for select. Model pipeline reads the rebuilt data pipeline's output (`hours.csv`, `windows/manifest.json`) | (this PR) |
| 2026-11-14 | Dhruv Puri | Data pipeline rebuilt for the window model: 10-s 30 Hz windows from the more affected wrist with an orientation fix, every diary hour kept, dose events and a patients table, 18 modules down to 8. Run on the model team's 10 participants and pushed as a DVC data version | (this PR) |
| 2026-11-14 | Shaikh Basim | Model pipeline (P5): the GCP project as Terraform in `infra/` (buckets, Workload Identity Federation for CI, Artifact Registry, Cloud SQL, the hosted MLflow on Cloud Run), the run command through every step (train, evaluate, bias, explain, select, release gate, export, register), the `model.yml` CI workflow, email alerts, and the shared DVC remote | (this PR) |
| 2026-11-13 | Ayush Varma | Model pipeline (P4): model selection, the release gate against production, the export bundle, and the registry with rollback. Placeholders for the phone runtime, the registry location and the Backend endpoint | (this PR) |
| 2026-11-11 | Sam Selvaraj | Model: the design decisions in MODEL.md. Model pipeline core: targets (kinesia score and tremor, now and next hour), the Dataset contract with a versioned window spec, the tracker contract with in-memory and MLflow trackers, and production logging. Data pipeline: dose events with LEDD from Jost et al. 2023. Fixed the Dependabot alerts | (this PR) |
| 2026-10-25 | Ayush Varma | Data pipeline run on 5 sample participants with `dvc repro` and Airflow, first DVC data version, results, bias findings, Gantt chart and screenshots in the docs, and this README updated | (this PR) |
| 2026-10-24 | Sam Selvaraj | Data pipeline: DVC, 7× faster processing, robustness fixes, CI, and tests (coverage 92%) | #5 |
| 2026-10-22 | Shaikh Basim | Data pipeline: fixed the run and the Airflow setup | #4 |
| 2026-10-21 | Tejaswini Sarikonda | Data pipeline: download, features, cleaning, validation, anomalies, bias, DAG, DVC stage, Docker and tests | #3 |
| 2026-10-12 | Sam Selvaraj | Repo setup: README, ARCHITECTURE.md, CI checks (secret scan, data file guard, Markdown lint) and the `dev` branch flow | — |
