# Model Pipeline

Trains the Parkinson's off-risk model that runs on the phone. It reads the data pipeline's hourly table, trains and tunes models with participant-grouped splits, checks them for bias, and packages the best deployable one as a TFLite bundle for the Backend's model registry.

> **Status:** the run command goes through every step, logged to MLflow, with alerts, and runs on GitHub Actions against the team's GCP project. Selection, the release gate, export and the registry (P4) are built; loading, training, evaluation and bias are stand-ins in `run_model_pipeline.py` until their owners merge them.

## Running

With Python 3.11 (or 3.12), from `Model_Pipeline/`:

```bash
pip install -r requirements.txt
pytest -q                               # synthetic data only, nothing downloaded
python -m scripts.run_model_pipeline --no-mlflow   # every step; stand-ins for the unbuilt ones
```

Or with only Docker: `docker compose up -d mlflow` for a local MLflow at <http://localhost:5001>, then `docker compose run --rm train` for the pipeline and `docker compose run --rm tests` for the tests.

The run command's options:

| Option | What it does |
|---|---|
| `--config <file>` | Another config instead of `config/model.yaml` |
| `--no-mlflow` | Keeps tracking in memory, so no MLflow server is needed |
| `--no-register` | Checks every gate and builds the bundle, but releases nothing. CI uses it for every run except a push to `dev` or `main`. |

The training data is the data pipeline's output, versioned by DVC. Get it with `dvc pull` in `../Data_Pipeline` (see [Training data](#training-data-dvc)). `paths.features_csv` in `config/model.yaml` points at it.

## Running in the cloud

Everything runs in the GCP project `team17-mlops-neu`, region `us-central1`. It's all defined as Terraform in [`../infra/`](../infra/), so `terraform plan` there shows what exists.

| What | Where |
|---|---|
| Training data (DVC remote) | `gs://team17-mlops-neu-training-data/cops-dvc` |
| MLflow server | Cloud Run service `mlflow`, private |
| MLflow runs, parameters and metrics | Cloud SQL instance `mednova-db`, database `mlflow` |
| MLflow files: plots, reports, bundles of each run | `gs://team17-mlops-neu-models/mlflow` |
| Released bundles | `gs://team17-mlops-neu-models/parkinsons-risk/v<N>/`, once `paths.registry_dir` points there. It's a local folder for now. |
| Images | Artifact Registry `ml`: `mlflow` (the server, from `mlflow/Dockerfile`) and `training` (this folder's image, one per release) |

To use any of it, a teammate needs Editor on the project, and to run once:

```bash
gcloud auth login
gcloud auth application-default login     # the login DVC and MLflow's Python client use
gcloud config set project team17-mlops-neu
```

### Opening the hosted MLflow

The server only lets in signed-in Google accounts, so a browser can't open its URL directly (403). Open it through a proxy that signs each request as you:

```bash
gcloud run services proxy mlflow --region=us-central1 --port=5001
```

Then open <http://localhost:5001>, and leave the command running while you use it.

- If the local MLflow from `docker compose` is running, it already has port 5001: use `--port=5002`, or stop it with `docker compose stop mlflow`.
- The server sleeps when unused, so the first page takes about 30 seconds.
- To log runs to it from your laptop, set `MLFLOW_TRACKING_URI=http://localhost:5001` while the proxy runs.

### CI: the model workflow

[`.github/workflows/model.yml`](../.github/workflows/model.yml) runs the pipeline on GitHub Actions:

1. Installs Python 3.11 and `requirements.txt`.
2. Signs in to GCP as the `training` service account through Workload Identity Federation, so there's no key file, and gets an ID token for the hosted MLflow.
3. Runs `python -m scripts.run_model_pipeline`, logged to the hosted MLflow. A failed gate fails the job.
4. Uploads `artifacts/` and `logs/` as the run's `model-run` artifact, even when it fails.
5. On a push to `dev` or `main` only: releases the model, and pushes the training image as `ml/training:<commit>`.

**For now it only runs by hand:** Actions → Model → Run workflow. That button shows once the workflow is on `main`. The triggers for every pull request and push are in the file, commented out, until the real steps are merged.

### Settings

In the repo's Settings → Secrets and variables → Actions. Changing them needs repo admin.

| Name | Kind | Value |
|---|---|---|
| `GCP_PROJECT_ID` | Variable | `team17-mlops-neu` |
| `GCP_WIF_PROVIDER` | Variable | `projects/818790524724/locations/global/workloadIdentityPools/github/providers/mlmodel` |
| `GCP_SERVICE_ACCOUNT` | Variable | `training@team17-mlops-neu.iam.gserviceaccount.com` |
| `MLFLOW_TRACKING_URI` | Variable | `https://mlflow-pnpgeaktha-uc.a.run.app`, with no trailing slash: it's also the ID token's audience |
| `ALERT_EMAIL` | Secret | Who gets the alerts, comma-separated |
| `SMTP_HOST`, `SMTP_PORT` | Secret | `smtp.gmail.com`, `587` |
| `SMTP_USER`, `SMTP_PASSWORD` | Secret | The sending Gmail account and its app password |

The GCP values are variables, not secrets: the workflow reads them with `vars.`, and none of them grants access on its own. There's no GCP key anywhere; GCP checks GitHub's ID token instead, and only accepts it from this repo.

To send alerts from your own machine, put the five alert settings in `Model_Pipeline/.env`, which Git ignores and `docker compose` reads.

### Alerts

[`scripts/alerts.py`](scripts/alerts.py) emails the team. Every subject starts with `[MedNOVA model]`, and every email links to the GitHub Actions run, or names the machine for a local run.

| When | Sent by | Subject |
|---|---|---|
| A gate fails | the run command | `The production gate failed`, with the reason, such as `cv/state_now/off_auc: 0.500 vs 0.500` |
| The run crashes on an unexpected error | the run command | `Run crashed: <error>`, with the stack trace |
| A model is released | the run command | `parkinsons-risk v<N> released` |
| Any other CI step fails: install, GCP sign-in, image push | `model.yml` | `CI job failed` |

A run that only checks (`--no-register`) and passes sends nothing. An alert that can't be sent is logged and never fails the run. Slack works the same way with `SLACK_WEBHOOK_URL`, but the team uses email.

### Training data (DVC)

The data pipeline's output is versioned by DVC: `../Data_Pipeline/dvc.lock` in Git says which version, and the files are in the bucket. From `../Data_Pipeline`, signed in as above:

```bash
pip install "dvc[gs]"     # or the data pipeline's requirements.txt
dvc pull                  # downloads the version dvc.lock names, including data/processed/hourly_features.csv
```

To publish a new version after changing the data pipeline: `dvc repro`, `dvc push`, then commit `dvc.lock`. An Airflow run pushes by itself; the commit is always by hand. See [Data_Pipeline/README.md](../Data_Pipeline/README.md), "Making a version".

## Code structure

```text
scripts/
├── core/                  contracts only, no heavy libraries
│   ├── targets.py         what can be predicted: OFF, tremor, dyskinesia
│   ├── data.py            Dataset: named input groups, labels, participants, slices
│   ├── model.py           the model contract and its predictions
│   ├── tracker.py         the tracking contract
│   └── errors.py          one error type per kind of failure
├── models/                one file per model family
│   ├── registry.py        family name to class, built from the config
│   └── composite.py       several models presented as one
├── data/
│   ├── loaders.py         the data pipeline's outputs into a Dataset
│   └── folds.py           the fixed test set and participant-grouped folds
├── tracking/
│   ├── mlflow_tracker.py  MLflow, with a local fallback
│   └── memory_tracker.py  for tests
├── steps/                 one class per step, in the order they run
├── utils/                 config, logging and file helpers
├── alerts.py              email (and Slack) alerts
└── run_model_pipeline.py  runs the steps from the config
tests/                     mirrors scripts/: tests/core, tests/models, ...
```

- **`core/` is what everything depends on.** It holds no MLflow, scikit-learn or deep learning code, so it stays light and portable. Change it only by team agreement.
- **Adding something is a new file, not an edit:** a model family in `models/`, a step in `steps/`, a tracker in `tracking/`.
- **Steps talk through their outputs,** never each other's internals, and raise a `core/errors.py` error when they can't go on, so the run is marked failed and an alert goes out.

## Who builds what

Letters are the components in the team plan.

| Owner | Components | Files |
|---|---|---|
| P1 | A Data loader, B Splits, F Validation + gate | `data/loaders.py`, `data/folds.py`, `steps/evaluate.py` |
| P2 | C Models + training, D Tuning, E Experiment tracking, G Sensitivity (hyperparameters) | `core/`, `models/`, `tracking/`, `steps/train.py`, `steps/tune.py`, `steps/sensitivity_hparams.py` |
| P3 | H Bias detection + mitigation, G Sensitivity (SHAP, ablation) | `steps/bias.py`, `steps/explain.py` |
| P4 | I Selection after bias, J Export + bundle, K Registry + rollback | `steps/select.py`, `steps/export.py`, `steps/compare_production.py`, `steps/register.py` |
| P5 | L Docker + orchestration, M CI/CD, N Alerts, O Infrastructure | `run_model_pipeline.py`, `alerts.py`, Docker and CI files |

All paths are under `scripts/`.

## Interfaces

The four hand-offs from the team plan. Build against these, with stand-ins for a teammate's part until it's merged. Change one only by agreement, in this file first.

### 1. `load_dataset(cfg)` → `X, y, groups, slices` (P1 → P2, P3)

In `scripts/data_loader.py`. Reads `paths.features_csv`. The four outputs share one index, one row per participant-hour.

- `X`: DataFrame of the `data.features` columns, without the `W_` prefix and in that order. That's the phone's input order.
- `y`: Series of 0 and 1. 1 when `kinesia_class` is `off`.
- `groups`: the `participant_id` of each row, for the participant-grouped splits.
- `slices`: DataFrame of the `bias.slice_columns`.

### 2. Predictions file (P2 → P1, P3)

`artifacts/predictions.csv`, one row per prediction:

| Column | Meaning |
|---|---|
| `participant_id`, `hour_index` | Which hour. `hour_index` joins back per-hour slices like `visit`. |
| `fold` | The cross-validation fold, or `test` for the held-out participants |
| `model` | The model's name, as logged in MLflow |
| `y_true` | 1 for off, 0 otherwise |
| `y_score` | Off risk, from 0 to 1 |

The team plan's version had a `wrist` column. It's gone, since the data pipeline now uses one wrist.

### 3. Model wrapper (P2 → P4)

In `scripts/models.py`, one class per model family, each with:

- `family`: `logistic`, `mlp` or `lightgbm`.
- `fit(X, y, sample_weight=None)`, returning itself. `sample_weight` is how bias mitigation by reweighting reaches training.
- `predict_proba(X)`: an array of shape (n, 2), as in scikit-learn. Column 1 is the off risk.
- `to_keras()`: a Keras model with 7 inputs and 1 sigmoid output that gives the same scores. It takes raw feature values, so any scaling is inside it, because the phone sends raw values. LightGBM doesn't convert to TFLite, so its `to_keras()` raises `NotImplementedError`. It's a benchmark only.

### 4. Bundle folder (P1, P3 → P4 → CI)

`artifacts/bundle/`, which CI pushes to `gs://<project-id>-models/parkinsons-risk/v<N>/`:

| File | From | What it holds |
|---|---|---|
| `model.tflite` | P4 | The exported model |
| `metadata.json` | P4 | The Backend's model entry: `family`, `model_id`, `version`, `feature_spec_version`, `inputs`, `output` (`off_risk`, `nudge_above`) and `test_cases` |
| `stats.json` | P4 | The training inputs' statistics, for drift. The Backend stores it as `<model_id>.stats.json`. |
| `metrics.json` | P1 | Held-out metrics, PR-AUC first |
| `bias.json` | P3 | Metrics per slice, the recall gap, and the mitigation result |

The Backend works out the model file's sha256 when it registers the model.

## Adding your step

- Fill in your module in the structure above, and put its settings in `config/model.yaml` under your own section. No hard-coded values.
- Test it in the matching folder, e.g. `tests/steps/test_bias.py` for `scripts/steps/bias.py`, with `tests/helpers.py`: `make_config(tmp_path)` and `make_features_table()` give a synthetic table with the real file's columns. Tests never read the real data.
- Add libraries under your heading in `requirements.txt`, pinned to an exact version.
- Don't import from `../Data_Pipeline`. Both folders have a package named `scripts`, so copy what you need, as `scripts/utils/` and `scripts/alerts.py` were.
- Everything a run writes goes in `artifacts/`, which Git ignores. Models and data never go in Git.
