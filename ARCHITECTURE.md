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

- **Folds:** 5 folds, fixed by hand. Each holds out 2 test patients who are never used for training or for any choice.
- **Early stopping:** inside each fold, one of the remaining patients is the validation patient. The rest are for training.
- **Normalisation:** statistics come from training patients only.
- **Hyper-parameters:** fixed before the run, never tuned on test patients.
- **Baselines on the same folds:** predict the training average, activity features, signal features, and harnet embeddings alone, each with logistic regression (tremor) and ridge regression (akinesia).
- **Leakage check:** the same model trained on shuffled labels must fall to chance.
- **Metrics:**
  - Fold AUC: any-tremor AUC inside each fold, averaged over folds. This is the main metric because it isn't affected by calibration differences between folds.
  - Pooled AUC: all test patients scored together, with a 95% confidence interval from resampling patients (1,000 resamples).
  - Within-patient tracking: Spearman correlation between the prediction and the diary for each patient, averaged over patients whose diary varies.
  - Non-DBS patients are reported separately.
- **Seeds:** the model runs with 3 seeds, and the spread between seeds is reported.

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

_Update · 2026-11-05_

**Decision:** While MJFF access is pending, we build and evaluate on **COPS** ([osf.io/5xvwn](https://osf.io/5xvwn/)): people with Parkinson's wearing wrist sensors at home, with an hourly patient diary of tremor and akinesia. The data pipeline in `Data_Pipeline/` processes COPS, and the model was evaluated on 10 COPS patients. PADS was used to choose the window encoder (section 3). MJFF is still the target for doctor-rated labels.
**Why:** COPS is open and has the home recordings and symptom labels we need now. Waiting for MJFF would have blocked both pipelines.

## 2. Preprocessing

_Decided · Owner: TBD · Date: 2026-10-20_

The datasets use different devices and sampling rates, and none matches our watch, so everything goes through one shared preprocessing step.

| Choice | Options | Notes |
|---|---|---|
| Common sampling rate | 30 Hz, 50 Hz, or 100 Hz | 50 Hz is the lowest rate in our datasets. 30 Hz is the rate the harnet10 encoder was pretrained on. |
| Window length | 10 s, or other | 10 s matches PADS task length, the planned per-window output, and harnet10's input |
| Axes and units | Convert all to the same orientation and units | Needs each device's documentation |

**Decision:** Resample to **30 Hz** and cut **10-second windows** (3 axes × 300 samples, so one hour is 360 windows). Use the more affected wrist, in g. Subtract each window's mean on every axis, which removes gravity and the watch's resting orientation.
**Why:** The model uses the frozen harnet10 encoder (section 3), which was pretrained on 30 Hz, 10-second windows, so its input has to match. All our datasets record at 50 Hz or more, so 30 Hz only ever downsamples.
**Not done yet:** `Data_Pipeline/` builds hourly features from the 100 Hz data. The 30 Hz window cache that the model trains on (raw windows plus harnet10 embeddings) is built by a script that isn't in this repo yet. Axes and units for MJFF are still to be checked once we have access.

## 3. Model approach

_Decided, accuracy still being assessed · Owner: TBD · Date: 2026-10-20_

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

**Decision:** The hybrid encoder with attention pooling is the main model. harnet10 + logistic regression is the baseline every new model has to beat.
**Why:**

- It's the only model that tracks tremor within a patient's day. The PD Monitor timeline needs that.
- The shuffled-label run still reaches +0.10 within-patient, so the real gain over chance is about +0.09.
- On PADS (Parkinson's vs healthy) the hybrid scored AUC 0.790–0.795, against 0.774 for harnet10 alone and 0.774 for the trained branches alone.

**Limits of this result:**

- With 10 patients the confidence intervals overlap heavily. Treat this as a direction, not proof.
- harnet10 alone is ahead on pooled AUC and tied for non-DBS patients.
- Akinesia doesn't work yet; activity features alone do better.
- Early stopping relies on one validation patient, and in 5 of the 15 runs it picked the first epoch.
- Attention pooling hasn't been compared with a plain average of the windows.

The full design, training details and open questions are in `Model Architecture.md`, which will be added to this repo together with the training code.

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

## 5. Data versioning

_Owner: TBD · Decide by: TBD_

| Option | Good | Bad |
|---|---|---|
| DVC + Cloud Storage | Ties each model to an exact data snapshot. Works with Git. | One more tool to learn |
| Cloud Storage object versioning only | Nothing new to learn | No link between code version and data version |

**Decision:** _TBD_
**Why:** —

## 6. Experiment tracking and model registry

_Owner: TBD · Decide by: TBD_

| Option | Good | Bad |
|---|---|---|
| MLflow | Taught in the course. Tracking and registry in one tool. | We host it ourselves |
| Vertex AI Experiments + Model Registry | Managed on GCP | Tied to GCP. Costs money. |
| Weights & Biases | Best dashboards | External service. Check data rules first. |

**Decision:** _TBD_
**Why:** —

## 7. Data validation

_Owner: TBD · Decide by: TBD_

| Option | Good | Bad |
|---|---|---|
| Great Expectations | Readable reports. Widely used. | Verbose setup |
| TFDV | Covered in the course labs. Detects drift. | Built around TensorFlow |
| pandera | Lightweight, plain Python | Fewer reports and dashboards |

**Decision:** _TBD_
**Why:** —

## 8. Where the model runs (all repos)

_Owner: TBD · Decide by: TBD_

| Option | Good | Bad |
|---|---|---|
| On the server | Easy to monitor, update and roll back. Shows the MLOps work well. | Raw movement data must be uploaded, which costs battery and bandwidth |
| On the phone or watch (TFLite) | Less data uploaded. More private. | Harder to update and monitor |

Input for this decision: the chosen model (section 3) includes harnet10, an 11M-parameter encoder, and takes about 15 ms per 10-s window on a CPU, mostly in harnet10. That leans toward the server.

**Decision:** _TBD_
**Why:** —

## 9. Drift monitoring

_Owner: TBD · Decide by: TBD_

| Option | Good | Bad |
|---|---|---|
| Evidently | Ready-made drift reports | One more service to run |
| Vertex AI Model Monitoring | Managed | Only works with Vertex AI serving |
| Our own statistics in the pipeline | Full control | More work |

**Decision:** _TBD_
**Why:** —

## 10. CI checks

_Decided · Owner: Sam Selvaraj · Date: 2026-10-12_

**Decision:** GitHub Actions runs a secret scan, a data file guard and Markdown lint on every PR and every push to `dev` or `main`. Language checks get added once the stack is chosen.
**Why:** Everyone with access to a repo sees everything in it, and the project handles licensed health data, so a leaked key or committed data is the most costly mistake we could make. These checks don't depend on the undecided stack. GitHub Actions is built into GitHub and covered in the course.
