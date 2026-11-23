# MLModel Architecture

How the data and model pipelines are built: the options we considered for each choice and why we picked the one we did. When a decision is made, fill in **Decision**, **Why**, the owner and the date. To change a decision later, add a new dated entry under it instead of rewriting it.

## Where it fits

```text
Datasets ──▶ Data pipeline ──▶ Model pipeline ──▶ Model registry ──▶ Backend serves it
                                                                       │
                          Live data from patients ◀────────────────────┘
                          (monitored for drift, can trigger retraining)
```

## Fixed rules

- Train/test splits are by **patient**. No patient appears in both, or the scores will be inflated.
- A new model replaces the current one only if it does better on the same folds as the evaluation protocol below, on both **fold AUC for any tremor** and **within-patient tremor tracking**, and is no worse on pooled AUC.
- No data, features or model weights in Git.

## Evaluation protocol

Every model and baseline is scored the same way, so their numbers can be compared.

_Updated 2026-11-15 to the protocol the model pipeline runs. It replaces the first plan of 5 folds with 2 test patients each, which was made for the 15-patient prototype._

- **Patients:** of the 66 COPS patients, 10 are test patients, 6 are kept for the app's demo, and 50 are for development. The split is fixed in `Data_Pipeline/data/processed/patient_split.json` (DVC), with the rule that picked it. The demo patients are never in the model's data.
- **Folds:** the 50 development patients in 2 folds of 25 (`cv_folds`). Each fold trains on one half and is scored on the other, so every development patient is scored once, by a model that never saw them.
- **Test patients:** scored once, at the end, by the final models (`train.score_test`). Never used for training, tuning or any choice.
- **Early stopping:** on the fold's validation patients.
- **Normalisation:** the cohort statistics come from training patients only. Each patient's features are also compared with their own past 7 days (the patient baseline, in [Model_Pipeline/README.md](Model_Pipeline/README.md#what-changed-the-scores)), which only ever looks back.
- **Hyper-parameters:** tuned with Optuna on the development folds, never on test patients.
- **Baselines on the same folds:** `window_logistic` and `window_boosting`, which score each window on its own with the same harnet10 features and patient baseline as the transformer.
- **Leakage check:** boosting fitted to shuffled labels must score near chance (AUC within 0.4–0.6) on state and tremor, within patients.
- **Metrics:** every diary question as an AUC, pooled (all hours together) and within patients (each patient's hours against each other, averaged over patients). Evaluate adds a 95% interval for each pooled AUC from resampling patients. Selection uses within-patient OFF AUC (`cv/state_now/off_auc_within`), the score that matters for telling one patient's OFF hours from their good ones.
- **Bias:** the same AUCs per patient group (sex, age group, visit, DBS, Hoehn & Yahr, PD subtype), judged on groups of 5 or more patients.
- **Seeds:** a list in the config. The 2026-11-15 run used one seed (42); the folds give the spread.

---

## 1. Datasets

_Decided · Owner: TBD · Date: 2026-10_

| Dataset | Role |
|---|---|
| PADS | Learn what Parkinson's wrist movement looks like. Open, so we can build the pipeline now. |
| MJFF Levodopa Response Study | Train and test the severity model on doctor ratings |
| Parkinson@Home | Check the model works in real homes |

Ruled out: BEAT-PD (no working download) and PD-BioStampRC21 (too few patients with tremor).

**Decision:** PADS + MJFF Levodopa, with Parkinson@Home for home testing. Fall back to PADS + Parkinson@Home if MJFF access isn't granted in about 2 weeks.
**Why:** It's the only combination with doctor-rated severity across a full medication dose cycle.

_Note · 2026-11-05_

While MJFF access is pending, we build and evaluate on **COPS** ([osf.io/5xvwn](https://osf.io/5xvwn/)): people with Parkinson's wearing wrist sensors at home, with an hourly patient diary of tremor and akinesia. The data pipeline in `Data_Pipeline/` processes COPS, and the model was evaluated on 10 COPS patients. PADS was used to choose the window encoder (section 3). MJFF is still the target for doctor-rated labels. COPS is open and has the home recordings and symptom labels we need now, so waiting for MJFF would have blocked both pipelines.

## 2. Preprocessing

_Owner: TBD · Decide by: TBD_

The datasets use different devices and sampling rates, and none matches our watch, so everything goes through one shared preprocessing step.

| Choice | Options | Notes |
|---|---|---|
| Common sampling rate | 30 Hz, 50 Hz, or 100 Hz | 50 Hz is the lowest rate in our datasets. 30 Hz is the rate the harnet10 encoder was pretrained on. |
| Window length | 10 s, or other | 10 s matches PADS task length, the planned per-window output, and harnet10's input |
| Axes and units | Convert all to the same orientation and units | Needs each device's documentation |

**What the prototype uses:** 30 Hz and 10-second windows (3 axes × 300 samples, so one hour is 360 windows), from the more affected wrist, in g. It subtracts each window's mean on every axis, which removes gravity and the watch's resting orientation. The reason is that the leading model candidate (section 3) uses the frozen harnet10 encoder, which was pretrained on 30 Hz, 10-second windows. All our datasets record at 50 Hz or more, so 30 Hz only ever downsamples.

**Decision:** _TBD_
**Why:** —

**Built (2026-11-15):** the data pipeline cuts the windows itself (`Data_Pipeline/data_pipeline/sensor.py`): 30 Hz, 10 s, one wrist, mean removed, the watch's orientation decided per wear bout, and one wrist mirrored so both read alike; harnet10's features are cached by the model pipeline. The Backend cuts live watch data with the same code (Backend `app/scoring/inputs.py`), so a model scores live data as it was trained. Axes and units for MJFF are still to be checked once we have access.

## 3. Model approach

_Owner: TBD · Decide by: TBD_

| Option | Good | Bad |
|---|---|---|
| Hand-made features (tremor-band power, movement statistics) + gradient boosting | Fast. Explainable. Works with little data. | Accuracy is limited by the features |
| 1D CNN on raw windows | Learns its own features | Needs more data. Harder to explain. |
| Pretrain on PADS, fine-tune on MJFF | Uses PADS's much larger group of people to make up for MJFF's small one | More complex training |
| Pretrained encoder (harnet10, frozen) + logistic regression | Wrist-movement knowledge from UK Biobank (700k person-days). Simple. | One score per hour, so no timeline through the day |
| **Hybrid encoder + attention pooling over windows** | Scores every 10-s window, so it can draw a timeline. Learns from hourly labels. | More complex. Only a small edge over harnet alone so far. |

The hybrid encoder joins frozen harnet10 embeddings with a trained 1D CNN on the raw signal and a trained 2D CNN on the spectrogram (0–15 Hz). Heads predict P(tremor ≥ 1), P(tremor ≥ 2) and akinesia 0–3 for every window. Gated attention weights the windows, and the hour prediction is the weighted average of the window predictions. This is multiple-instance learning: the diary labels the hour, not the window.

We started with feature baselines as planned (activity and signal features with logistic regression; gradient boosting wasn't tried) and compared the others against them using the [evaluation protocol](#evaluation-protocol). Results on 10 COPS patients, test patients never seen in training:

| Model | Fold AUC | Pooled AUC [95% CI] | Within-patient tremor | Within-patient akinesia | Non-DBS fold AUC |
|---|---|---|---|---|---|
| Training average | 0.500 | 0.314 [0.16–0.51] | n/a | n/a | 0.500 |
| Activity features | 0.589 | 0.606 [0.51–0.69] | +0.08 | **+0.11** | 0.512 |
| Signal features | 0.639 | 0.720 [0.52–0.86] | +0.06 | +0.05 | 0.499 |
| harnet10 alone | 0.737 | **0.786** [0.63–0.90] | +0.11 | +0.02 | **0.620** |
| **Hybrid (3-seed average)** | **0.754** | 0.731 [0.49–0.93] | **+0.19** | −0.08 | 0.614 |
| Shuffled labels (leakage check) | 0.474 | 0.415 [0.23–0.63] | +0.10 | −0.05 | 0.519 |

**What the results suggest so far:** the hybrid encoder with attention pooling is the leading candidate, and harnet10 + logistic regression is the baseline to beat.

- The hybrid is the only model that tracks tremor within a patient's day. The PD Monitor timeline needs that.
- The shuffled-label run still reaches +0.10 within-patient, so the real gain over chance is about +0.09.
- On PADS (Parkinson's vs healthy) the hybrid scored AUC 0.790–0.795, against 0.774 for harnet10 alone and 0.774 for the trained branches alone.

**Limits of this result:**

- With 10 patients the confidence intervals overlap heavily. Treat this as a direction, not proof.
- harnet10 alone is ahead on pooled AUC and tied for non-DBS patients.
- Akinesia doesn't work yet; activity features alone do better.
- Early stopping relies on one validation patient, and in 5 of the 15 runs it picked the first epoch.
- Attention pooling hasn't been compared with a plain average of the windows.

The full design, training details and open questions are in `Model Architecture.md`, which will be added to this repo together with the training code.

**Decision:** _TBD_
**Why:** —

## 4. Pipeline orchestration

_Owner: TBD · Decide by: TBD_

| Option | Good | Bad |
|---|---|---|
| Airflow (self-hosted in Docker) | Taught in the course. Standard. | Heavy to run |
| Cloud Composer (managed Airflow) | No Airflow upkeep | Costly to keep running |
| Vertex AI Pipelines | Managed and built for ML on GCP | Learning curve |
| TFX | Listed in the syllabus. Validation built in. | Built around TensorFlow. Steep. |

**Decision:** _TBD_
**Why:** —

_Decided · Owner: Sam Selvaraj · Date: 2026-11-18_

**Decision:** Airflow, self-hosted in Docker: one Airflow for both pipelines ([`airflow/`](airflow/README.md)), on a training VM that CI makes for each run and that deletes itself after. Each task runs one step in its pipeline's own image, so Airflow never imports either pipeline. CI starts every run (a merge that bumps `Model_Pipeline/config/retrain.yaml`, a run by hand, or a drift retrain), so training is triggered from CI, as the model pipeline's assignment asks.
**Why:** The course teaches Airflow and the data pipeline's assignment asks for it, and its UI shows every step's status, duration and log, next to MLflow's scores. Composer costs a few hundred dollars a month even idle; a VM that exists only while it trains costs only then.

## 5. Data versioning

_Owner: TBD · Decide by: TBD_

| Option | Good | Bad |
|---|---|---|
| DVC + Cloud Storage | Ties each model to an exact data snapshot. Works with Git. | One more tool to learn |
| Cloud Storage object versioning only | Nothing new to learn | No link between code version and data version |

**Decision:** _TBD_
**Why:** —

**As built (2026-11-18):** DVC with the team's Cloud Storage bucket (`Data_Pipeline/dvc.yaml`); CI opens a PR with each new data version's `dvc.lock`.

## 6. Experiment tracking and model registry

_Owner: TBD · Decide by: TBD_

| Option | Good | Bad |
|---|---|---|
| MLflow | Taught in the course. Tracking and registry in one tool. | We host it ourselves |
| Vertex AI Experiments + Model Registry | Managed on GCP | Tied to GCP. Costs money. |
| Weights & Biases | Best dashboards | External service. Check data rules first. |

**Decision:** _TBD_
**Why:** —

**As built (2026-11-18):** MLflow for tracking, hosted on Cloud Run (`infra/mlflow.tf`); the registry is a folder per version in the models bucket, with `active.json` naming the one in use (`Model_Pipeline/model_pipeline/steps/register.py`).

## 7. Data validation

_Owner: TBD · Decide by: TBD_

| Option | Good | Bad |
|---|---|---|
| Great Expectations | Readable reports. Widely used. | Verbose setup |
| TFDV | Covered in the course labs. Detects drift. | Built around TensorFlow |
| pandera | Lightweight, plain Python | Fewer reports and dashboards |

**Decision:** _TBD_
**Why:** —

**As built (2026-11-18):** the data pipeline's own checks (`Data_Pipeline/data_pipeline/checks.py`): a contract that stops the run, warnings, and the bias report.

## 8. Where the model runs (all repos)

_Owner: TBD · Decide by: TBD_

| Option | Good | Bad |
|---|---|---|
| On the server | Easy to monitor, update and roll back. Shows the MLOps work well. | Raw movement data must be uploaded, which costs battery and bandwidth |
| On the phone or watch (TFLite) | Less data uploaded. More private. | Harder to update and monitor |

Input for this decision: the leading model candidate (section 3) includes harnet10, an 11M-parameter encoder, and takes about 15 ms per 10-s window on a CPU, mostly in harnet10. That leans toward the server.

**Decision:** _TBD_
**Why:** —

_Decided · Owner: Sam Selvaraj · Date: 2026-11-13, built 2026-11-18_

**Decision:** On the server, in the Backend (Backend `app/scoring/`): each hour, it scores every Parkinson's patient's new hours with the version the registry's `active.json` names, with the model pipeline's own code, and stores the predictions the care portal shows. Any model family fits: the Backend builds the inputs the saved model lists.
**Why:** MODEL.md, "On the phone", update of 2026-11-13: the model is built and checked in Python, and the server runs that exact code; the transformer's state would be the hard part of a phone export. On CPU, v1 scores an hour of one patient's watch data in about 1.5 s.

## 9. Drift monitoring

_Owner: TBD · Decide by: TBD_

| Option | Good | Bad |
|---|---|---|
| Evidently | Ready-made drift reports | One more service to run |
| Vertex AI Model Monitoring | Managed | Only works with Vertex AI serving |
| Our own statistics in the pipeline | Full control | More work |

**Decision:** _TBD_
**Why:** —

_Decided · Owner: Sam Selvaraj · Date: 2026-11-18_

**Decision:** Our own statistics, in the Backend: prediction drift. Each night the Backend compares the P(OFF) the released model gave every worn hour with how often it said each P(OFF) band on its out-of-fold training hours (the bundle's `stats.json` `prediction_reference`), as a PSI. Over 0.25 for 3 nights in a row, with 5+ patients, a report goes to GCS; [`drift.yml`](.github/workflows/drift.yml) opens a GitHub issue, and retrains if nobody closes it within 24 hours (at most once a week). The 7 hourly features' drift is shown, not acted on.
**Why:** It works for any model family, whatever its inputs, since every model gives P(OFF), and needs no extra service. The 7 hourly features aren't the models' inputs any more, so their drift is only a hint that the watch data changed.

## 10. CI checks

_Decided · Owner: Sam Selvaraj · Date: 2026-10-12_

**Decision:** GitHub Actions runs a secret scan, a data file guard and Markdown lint on every PR and every push to `dev` or `main`. Language checks get added once the stack is chosen.
**Why:** Everyone with access to a repo sees everything in it, and the project handles licensed health data, so a leaked key or committed data is the most costly mistake we could make. These checks don't depend on the undecided stack. GitHub Actions is built into GitHub and covered in the course.
