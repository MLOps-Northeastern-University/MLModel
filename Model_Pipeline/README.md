# Model Pipeline

Trains the model behind Careportal's symptom timeline. From the watch's movement, the doses and the check-ins, it estimates each hour whether the patient is asleep, their waking state (the diary's kinesia score, OFF to dyskinetic) and their tremor, now and in the next hour. It reads the data pipeline's outputs, trains and tunes with participant-grouped splits, checks for bias, and packages the chosen model as a bundle for the model registry. The model runs on the server, not the phone ([MODEL.md](../MODEL.md), "On the phone").

> **Status:** in progress.
>
> - **Built:** the contracts in `core/` (with the shared scores, `core/metrics.py`), the data loader, the patient split, three model families and the registry, training, tuning, hyperparameter sensitivity, the MLflow tracker, and selection, export, registry and the release gate (P4), and evaluate with the validation gate (P1).
> - **Built (P3):** the bias step (slices, gate, reweighing) and the explain step (Shapley values over input groups, ablation), tested on synthetic data and a small test model family.
> - **Wired (P5):** the run command goes through every step with alerts, and runs on GitHub Actions against the team's GCP project, logged to the hosted MLflow. It calls every real step in order: train, evaluate, bias, explain, select, compare, export, register.
> - **Still a docstring:** `models/composite.py`.
> - **Not run on the real data yet:** training, tuning and sensitivity are tested on synthetic data only.
>
> Each module says what it's for and who owns it.

## Running

With Python 3.13, from `Model_Pipeline/` (3.13 for PyTorch; the data pipeline stays on 3.11):

```bash
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build for a GPU
pip install -r requirements.txt
pytest -q                               # synthetic data only, nothing downloaded
python -m scripts.run_model_pipeline --no-mlflow   # every step, without an MLflow server
```

The run needs the data pipeline's outputs (`dvc pull` in `../Data_Pipeline`), the patient split (`paths.split_file`, made once with `python -m scripts.data.folds create`) harnet10's weights (`MODEL_HARNET_HUB`, below), and, while `train.use_tuned` is true, the tuned settings: run `python -m scripts.steps.tune` first. Without them it stops at train with a clear error.

Or with only Docker: `docker compose up -d mlflow` for a local MLflow at <http://localhost:5001>, then `docker compose run --rm train` for the pipeline and `docker compose run --rm tests` for the tests.

The run command's options:

| Option | What it does |
|---|---|
| `--config <file>` | Another config instead of `config/model.yaml` |
| `--no-mlflow` | Keeps tracking in memory, so no MLflow server is needed |
| `--no-register` | Checks every gate and builds the bundle, but releases nothing. CI uses it for every run except a push to `dev` or `main`. |

The training data is the data pipeline's output, versioned by DVC. Get it with `dvc pull` in `../Data_Pipeline` (see [Training data](#training-data-dvc)). `paths.features_csv` in `config/model.yaml` points at it.

### The steps P2 built

```bash
python -m scripts.data.folds create --test <ids> --rule "<how they were chosen>"   # once: the patient split
python -m scripts.data.folds add [--test <new ids>]                               # when patients are added
python -m scripts.steps.tune                  # tune the candidates in tune: (config/model.yaml)
python -m scripts.steps.train                 # cross-validate, refit and track every candidate
python -m scripts.steps.sensitivity_hparams   # which tuned settings mattered, with plots
```

Each logs to MLflow at `MLFLOW_TRACKING_URI` (else `mlflow.tracking_uri` in the config). To log to the team's server, see [Opening the hosted MLflow](#opening-the-hosted-mlflow).

Where things are, set by environment variables, not the config, because they differ per machine:

| Variable | What | Default |
|---|---|---|
| `MODEL_HARNET_HUB` | A local copy of OxWearables/ssl-wearables, with harnet10's weights (academic, non-commercial use only; never in Git or a bundle) | `~/.cache/torch/hub/OxWearables_ssl-wearables_main` |
| `MODEL_FEATURE_CACHE` | harnet10's features, computed once per patient | `artifacts/cache/harnet10` |
| `MODEL_PRETRAIN_CACHE` | The transformer's pretrained bodies, reused by fits with the same pretraining; `off` to disable | `artifacts/cache/pretrained` |
| `MODEL_DEVICE` | `cuda` or `cpu` | `cuda` when there is one |

Training works on CPU, much more slowly for the transformer. Both caches only save time: deleting them never changes a result.

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

1. Installs Python 3.13, PyTorch's CPU build and `requirements.txt`.
2. Signs in to GCP as the `training` service account through Workload Identity Federation, so there's no key file, and gets an ID token for the hosted MLflow.
3. Runs `python -m scripts.run_model_pipeline`, logged to the hosted MLflow. A failed gate fails the job.
4. Uploads `artifacts/` and `logs/` as the run's `model-run` artifact, even when it fails.
5. On a push to `main` only: releases the model, and pushes the training image as `ml/training:<commit>`.

**When it runs:** when `Model_Pipeline/**` or `Data_Pipeline/dvc.lock` changes on the way to `main`:

- a pull request into `main` (`dev` → `main` at release time) checks every gate and releases nothing;
- the merge into `main` trains, checks and releases;
- Actions → Model → Run workflow checks any branch, and releases nothing.

Work on `dev` doesn't start it; `checks.yml` tests every pull request. To run it for `dev` too, add `dev` to both `branches` lists in the workflow.

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
| Training finishes and a model is released | the run command | `parkinsons-risk v<N> released` |
| Training finishes, every gate passed, nothing released (`--no-register`) | the run command | `Training finished, every gate passed` |
| Any other CI step fails: install, GCP sign-in, image push | `model.yml` | `CI job failed` |

So every run sends exactly one email: how it finished, or why it stopped. An alert that can't be sent is logged and never fails the run. Slack works the same way with `SLACK_WEBHOOK_URL`, but the team uses email.

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
│   ├── targets.py         what is predicted: sleep, the waking state, tremor; now and next hour
│   ├── data.py            Dataset: named input groups, labels, participants, slices
│   ├── model.py           the model contract and its predictions
│   ├── metrics.py         the scores every step reports: pooled and within-patient (numpy only)
│   ├── tracker.py         the tracking contract
│   └── errors.py          one error type per kind of failure
├── models/                one module or package per model family
│   ├── window_classifier.py        window_logistic, window_boosting: each 10-s window on its own, no past
│   ├── motor_causal_transformer/   the window model: harnet10 features, attention over 8 h of past
│   ├── registry.py        family name to class; builds the config's candidates, loads saved models
│   └── composite.py       several models presented as one
├── data/
│   ├── loaders.py         the data pipeline's outputs into a Dataset
│   └── folds.py           the patient split: fixed test patients, folds over the rest
├── tracking/
│   ├── mlflow_tracker.py  MLflow; fails before training if the server can't be reached
│   └── memory_tracker.py  for tests
├── steps/                 one class per step, in the order they run
├── experiments.py         what train and tune share: cross-validation, scoring, the tuned-settings file
├── reporting/             charts (scorecard, confusion matrices, model comparison), saved and attached to runs
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
| P2 | C Models + training, D Tuning, E Experiment tracking, G Sensitivity (hyperparameters) | `core/`, `models/`, `tracking/`, `experiments.py`, `reporting/`, `steps/train.py`, `steps/tune.py`, `steps/sensitivity_hparams.py` |
| P3 | H Bias detection + mitigation, G Sensitivity (SHAP, ablation) | `steps/bias.py`, `steps/explain.py` |
| P4 | I Selection after bias, J Export + bundle, K Registry + rollback | `steps/select.py`, `steps/export.py`, `steps/compare_production.py`, `steps/register.py` |
| P5 | L Docker + orchestration, M CI/CD, N Alerts, O Infrastructure | `run_model_pipeline.py`, `alerts.py`, Docker and CI files |

All paths are under `scripts/`. `data/folds.py` was written by P2 as the stand-in training needed. P1 takes it over.

## Interfaces

The four hand-offs between owners. Build against these, with stand-ins for a teammate's part until it's merged. Change one only by agreement, in this file first.

> **Updated 2026-11-13 (P2), to agree with the team.** The first version described one model: a yes/no OFF score from the 7 hourly features, exported to the phone as TFLite. The model is now a window model on the server with three outputs (see [MODEL.md](../MODEL.md)), so all four hand-offs changed:
>
> - 1, 2 and 3 describe the code as it is (P1 and P3 to confirm 2);
> - 4 is a proposal, still open, marked **Open**.
>
> The 7 hourly features stay: they're the baselines' input and the drift statistics.

### 1. `load_dataset(cfg, groups)` → `Dataset` (P1 → P2, P3)

In `scripts/data/loaders.py`; the `Dataset` itself is in `scripts/core/data.py`.

```python
data = load_dataset(cfg, groups=("windows", "events", "patients"))
train, val = data.subset(train_ids), data.subset(val_ids)
```

Only the groups asked for are read. Every Dataset also has `hours` and `labels`:

| Part | What it holds | Who reads it |
|---|---|---|
| `hours` | One row per diary hour: `participant_id`, `hour_index`, and the bias slice columns | Everyone; P3 for the slices |
| `labels` | One column per label (see below): the class index, or NaN where there's no answer | Training, evaluate, bias |
| `features` | The 7 hourly features, one row per hour | The hourly baselines |
| `windows` | Per participant: the 10-s accelerometer windows, their start times and whether the watch was worn. Memory-mapped | The window models |
| `events` | One row per dose (LEDD, source) or check-in (the two answers) | The transformer |
| `patients` | One row per participant: more affected side, DBS | The transformer, and the no-DBS scores |

- **One clock for everything:** seconds since the participant's first diary day began. A window or event at `time_s` belongs to hour `time_s // 3600`.
- **The window cut is versioned.** `windows` in `config/model.yaml` gives the seconds, rate and axes, and a Dataset cut with another version is refused.
- **The patient split is a file** (`paths.split_file`, made with `scripts/data/folds.py`), never recomputed. The test patients never change, no one ever moves into them, and the folds spread the patients with DBS and with dyskinetic hours. It names patients, so it's kept with the data, never in Git.
- **A label is a target at a horizon,** for example `state_now` or `tremor_next_hour`. The targets are in `scripts/core/targets.py`:

| Target | Classes |
|---|---|
| `sleep` | 2: awake, asleep |
| `state` | 7: the waking kinesia score, −3 to +3. No label in sleep hours |
| `tremor` | 3: the tremor score, 0 to 2. No label in sleep hours |

  The broad states are sums of state classes: OFF is −3 to −1, ON is 0, dyskinetic is +1 to +3.

### 2. Predictions and scores (P2 → P1, P3)

**In code:** `Predictions` in `scripts/core/model.py`, which `model.predict(data)` returns.

- `hours`: `participant_id` and `hour_index`, one row per hour of the data, in its order.
- `probs`: per label, the probability of each class per hour. Each row adds up to 1.
- `windows`: for a window model, the same probabilities for every worn 10-s window. These feed the app's timeline.
- Helpers:
  - `p_asleep("sleep_now")`: the chance the patient is asleep;
  - `p_state("state_now", "off")`: the chance of OFF if awake;
  - `expected_score(label)`: the expected diary score;
  - `frame(label)`: one label's probabilities as a table.

**On disk:** train writes one file per label, `artifacts/predictions/<label>.csv`, because each target has a different number of classes. These files hold every candidate's out-of-fold hours (and the test patients' hours, the one time they're scored). They stay local and never go to MLflow, because they're per patient-hour.

| Column | Meaning |
|---|---|
| `participant_id`, `hour_index` | Which hour. `hour_index` joins back per-hour slices such as `visit` |
| `fold` | The cross-validation fold, or `test` for the held-out participants |
| `seed` | The training seed |
| `model` | The candidate's name, as logged in MLflow |
| `worn` | Whether the hour has a worn window. An hour without one gets the training class mix, so the scores leave it out |
| `y_true` | The true class index, empty where there's no answer |
| one column per class | Its probability, named as in `core/targets.py` (`severe_akinesia`, …) |

**Scores** are in `scripts/core/metrics.py`, shared by every step, and named `{split}/{label}/{metric}`, such as `cv/state_now/off_auc_within`. Every score comes **pooled** (all patients' hours together) and **within** (`_within`: each patient's own score, averaged over patients):

- AUCs: `off_auc`, `on_auc`, `dyskinetic_auc` (state), `asleep_auc` (sleep), `any_tremor_auc` (tremor), plus `off_auc_no_dbs` (patients without DBS, pooled only);
- the most likely class: `balanced_accuracy`, `macro_f1`, and per class `recall_<class>` and `precision_<class>` (pooled only);
- the diary score (state and tremor): `score_spearman` and `score_mae`.

`selection.metric` picks the model and is what tuning maximises. It's `cv/state_now/off_auc` (pooled) today. P2 proposes `off_auc_within`, which the team still needs to agree.

### 3. The model contract (P2 → P4)

Every model family subclasses `Model` in `scripts/core/model.py`. The base class checks inputs, outputs and train/validation overlap, so no family can skip those checks.

- **Class attributes:**
  - `family`: its name in the config and the registry;
  - `needs`: the Dataset groups it reads, such as `("windows", "events", "patients")`;
  - `scores_windows`: True if it also scores every worn window.
- **`fit(train, val=None, sample_weight=None, seed=0)`**, which returns the model. `val` is for early stopping and must have different participants. `sample_weight` (one weight per training hour) is how bias mitigation by reweighting reaches training. An hour of weight 0 counts as if it weren't there.
- **`predict(data)`** returns Predictions (interface 2) for every hour of `data`.
- **`save(folder)` / `Family.load(folder)`:** the weights plus a `manifest.json` (family, targets, settings, how it was trained, code and library versions). A model is never overwritten, and load refuses another family's folder. `models/registry.py`'s `load(folder)` opens any family's model by its manifest.
- **`default_params()`:** the settings and their defaults. Tuning's search spaces are in the config (`tune:`), not in code.

The families:

| Family | Needs | `sample_weight` | Status |
|---|---|---|---|
| `window_logistic` | windows | yes | Built. Baseline: logistic regression on each window's harnet10 features, an hour = the mean of its windows |
| `window_boosting` | windows | yes | Built. Baseline: the same with PCA and gradient boosting |
| `motor_causal_transformer` | windows, events, patients | yes (diary step) | Built, with 5-fold CV results in MODEL.md |
| Hourly baselines (training average, logistic regression on the 7 features) | features | – | Not built |

**Adding a family:** a `Model` subclass in `models/`, one line in `models/registry.py`'s `FAMILIES`, and a candidate in the config's `models.candidates`. Train, tune and sensitivity need no changes.

There's no `to_keras()` or phone export: the model runs on the server.

### Evaluate (P1)

`scripts/steps/evaluate.py` runs after train, on its `candidates.json` and predictions files, on the same hours train scores (labelled, worn, out of fold). Per candidate it adds a 95% interval for each pooled AUC by resampling patients (1,000 times, `evaluation` in the config), each question's AUC per fold and per patient (`evaluation.json`) and the ROC curves (`roc.png`), all in `artifacts/evaluation/<candidate>/`. It sets `gates_passed` to false when a metric in `validation_gate.min` is missing or below its value, so select skips that candidate.

### 4. Bundle folder (P1, P3 → P4 → CI) (Open)

`artifacts/bundle/`, which `register.py` copies into the registry as `<family>/v<N>/`. The proposal, now that the model runs on the server:

| File | From | What it holds |
|---|---|---|
| `model/` | P4 | The folder `model.save()` wrote: `manifest.json` and `weights/` (for the transformer: `network.pt` and `training.json`, with the harnet10 fingerprint). This replaces a phone model file |
| `manifest.json` | P4 | What the Backend needs to run and check the model: family, version, window spec version, labels, and the model folder's sha256 |
| `stats.json` | P4 | The training data's statistics for drift, starting with the 7 hourly features |
| `metrics.json` | P1 | The chosen candidate's `artifacts/evaluation/<name>/metrics.json`: train's `cv/` scores plus evaluate's 95% intervals (`_auc_lo`, `_auc_hi`), which the next run's release gate compares against |
| `bias.json` | P3 | Metrics per slice and the mitigation result: `artifacts/bias/bias.json` |

harnet10 isn't in the bundle: its licence limits redistribution. The server loads its own pinned copy, and the model refuses one whose sha256 differs from training.

Still open:

- where the server gets harnet10;
- where the registry lives (`paths.registry_dir`);
- the Backend endpoint for a new version;
- how `select.py`'s "exportable" check and `export.py`'s conversion change now that nothing is converted for the phone.

## Training, tuning and sensitivity (P2)

They share one cross-validation (`scripts/experiments.py`), so a tuning trial and a training run are scored exactly the same way: each fold's training patients fit, its validation patients scored, for every seed. The test patients are loaded only when `train.score_test` is true, once, at the end of model work.

| Step | Config | What it does | Writes |
|---|---|---|---|
| `tune` | `tune:` | One Optuna study (TPE, seeded) per listed candidate. Each trial is a cross-validation scored by `selection.metric`. Studies resume after a crash, and any change to the split, data, candidate or space starts a new one. Transformer trials that change only diary settings reuse each fold's pretrained body | `artifacts/tuning/` (studies, a trials table per candidate), `artifacts/tuned_params.json` |
| `train` | `models:`, `train:` | A shuffled-label check first (a model must score near chance on shuffled labels), then every candidate cross-validated, refit on all development patients, and saved. With `use_tuned`, a tuned candidate trains with its tuned settings, refused if they were tuned on another split, data or config | `artifacts/models/`, `artifacts/predictions/`, `artifacts/candidates.json`, `artifacts/charts/` |
| `sensitivity_hparams` | (reads the studies) | Each tuned setting's importance (fANOVA), plus plots: importance, score against each setting, score per trial | `artifacts/sensitivity/<candidate>/` |

In MLflow, each step is a run with a child per candidate. train and tune go on to a child per fold (and tune per trial before that), with the scores above, fit times and each epoch's curves. For the transformer that includes pretraining's skill per target on the validation patients, so one pretraining can be compared with another. Runs carry the split's hash and a data fingerprint, so only comparable runs get compared. For a live look at the tuning studies, `optuna-dashboard sqlite:///artifacts/tuning/studies.db`.

Charts (`train.charts`): every train run with predictions gets its own, under `charts/` in its MLflow artifacts and in `artifacts/charts/<candidate>/<run>/`:

- **Scorecard:** every score for every label, pooled and within patients side by side, with what guessing would score marked.
- **Confusion matrix per label:** for each diary answer, the share of its hours the model predicted as each class.
- **Comparison** (on the `train` run): the candidates side by side on the metrics in `train.charts.compare`, with each one's spread across folds.

Each fold run gets them (`folds: false` turns that off), each candidate's run gets them for all its folds together, and the final run gets them when the test patients are scored. MLflow's own chart view still compares any logged metric across runs. A chart that fails to draw is a warning, never a failed run.

## Bias and explanations (P3)

```bash
python -m scripts.steps.bias      # after evaluate: reads artifacts/candidates.json and artifacts/predictions/
python -m scripts.steps.explain   # after bias: explains each final model by its input groups
```

| Step | Config | What it does | Writes |
|---|---|---|---|
| `bias` | `bias:` | Joins each predicted hour to its patient's groups (`slice_columns`, from the hours table or else the patients table) and scores every group with Fairlearn's `MetricFrame`, using `core/metrics.py`'s scores plus each question's recall, precision and false-positive rate. Each check in `bias.checks` fails when the gap between the best and worst judged group is over its `max_gap`. A group is judged only with `min_slice_rows` hours and `min_slice_participants` patients. A failing candidate alerts and is retrained with reweighing as `<name>_reweighted`, a new candidate. No candidate passing raises `GateError` | `artifacts/bias/`: `bias.json`, `bias_report.md`, `plots/`; the updated `artifacts/candidates.json` |
| `explain` | `explain:` | Exact Shapley values over each model's input groups (movement, doses and check-ins, patient fields, hourly features) for P(OFF), P(any tremor) and P(asleep), and the AUC drop when one group is left out | `artifacts/explain/<candidate>/` (`shap.json`, `shap.png`, `ablation.png`), `explain_report.md` |

- **One candidates file, passed along.** Bias reads `artifacts/candidates.json` and writes it back with each candidate's `gates["bias"]` and `gates_passed`, which is true only when every gate passed, so select skips a candidate that failed bias. Evaluate's earlier `gates_passed` is kept as `gates["validation"]` and never overwritten. A `_reweighted` candidate, which evaluate never saw, is held to the same `validation_gate.min`.
- **Nothing judged blocks.** A candidate passes only when at least one check was judged and none failed. If every group is too small to judge, the gate fails and alerts rather than passing by default, and no reweighted retrain is made, since there's nothing to reweigh.
- **Out-of-fold predictions by default** (`bias.predictions: cv`), so every development patient counts and the test patients stay unseen.
- **Reweighing fixes imbalance, not weak signal.** It weights each group's answers as if group and answer were independent (Kamiran and Calders), with weights from each fold's training patients only. It closes a gap a model learned from who is usually OFF; it can't close one where the movement shows OFF less clearly for a group. The report shows both gaps and the selection metric's change, so the trade-off is visible.
- **Leaving a group out** shuffles it (movement within each patient, hourly features within each patient), removes it (events) or swaps it with another patient's (patient fields), repeated `explain.repeats` times.

## Adding your step

- Fill in your module in the structure above, and put its settings in `config/model.yaml` under your own section. No hard-coded values.
- Test it in the matching folder, e.g. `tests/steps/test_bias.py` for `scripts/steps/bias.py`, with `tests/helpers.py`: `make_config(tmp_path)` gives a config, `make_features_table()` a synthetic hourly table with the real file's columns, and `make_dataset(spec)` a small synthetic Dataset with windows, events and patients. Tests never read the real data.
- Add libraries under your heading in `requirements.txt`, pinned to an exact version.
- Don't import from `../Data_Pipeline`. Both folders have a package named `scripts`, so copy what you need, as `scripts/utils/` and `scripts/alerts.py` were.
- Everything a run writes goes in `artifacts/`, which Git ignores. Models and data never go in Git.
