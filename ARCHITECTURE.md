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
- A new model replaces the current one only if it does better on a fixed test set.
- No data, features or model weights in Git.

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

## 2. Preprocessing

_Owner: TBD · Decide by: TBD_

The datasets use different devices and sampling rates, and none matches our watch, so everything goes through one shared preprocessing step.

| Choice | Options | Notes |
|---|---|---|
| Common sampling rate | 50 Hz, or 100 Hz | 50 Hz is the lowest rate in our datasets, so nothing gets upsampled |
| Window length | 10 s, or other | 10 s matches PADS task length and the planned per-window output |
| Axes and units | Convert all to the same orientation and units | Needs each device's documentation |

**Decision:** _TBD_
**Why:** —

## 3. Model approach

_Owner: TBD · Decide by: TBD_

| Option | Good | Bad |
|---|---|---|
| Hand-made features (tremor-band power, movement statistics) + gradient boosting | Fast. Explainable. Works with little data. | Accuracy is limited by the features |
| 1D CNN on raw windows | Learns its own features | Needs more data. Harder to explain. |
| Pretrain on PADS, fine-tune on MJFF | Uses PADS's much larger group of people to make up for MJFF's small one | More complex training |

Start with the feature baseline, then compare the others against it.

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
