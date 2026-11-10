# Model Pipeline

Trains the Parkinson's off-risk model that runs on the phone. It reads the data pipeline's hourly table, trains and tunes models with participant-grouped splits, checks them for bias, and packages the best deployable one as a TFLite bundle for the Backend's model registry.

> **Status:** skeleton. The shared config, utilities, alerts, synthetic test data, the run command's step list and the folder structure below are in place. Each module says what it's for and who owns it; owners fill in their own.

## Running

With Python 3.11 (or 3.12), from `Model_Pipeline/`:

```bash
pip install -r requirements.txt
pytest -q                               # synthetic data only, nothing downloaded
python -m scripts.run_model_pipeline    # lists the steps until they're built
```

The training data is the data pipeline's output, versioned by DVC. Get it with `dvc pull` in `../Data_Pipeline` once the shared remote is set up. `paths.features_csv` in `config/model.yaml` points at it.

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
├── alerts.py              email and Slack alerts
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
