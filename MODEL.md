# Model

What the Parkinson's model is for, what it has to produce, the data and inputs it can use, and the decisions made about it. Keep this file current: when a decision is made, fill in its entry in [Decisions](#decisions) with the owner and date. To change one later, add a new dated entry under it instead of rewriting it.

Pipeline choices (orchestration, tracking, registry, deployment) are in [ARCHITECTURE.md](ARCHITECTURE.md). This file is about the model itself.

## 1. What the model is for

Careportal's patient dashboard shows symptoms through the day, next to the doses. Most of the dashboard comes from the app and the clinic. The model is needed only for the symptom estimates, which a patient can't reliably report hour by hour.

| Dashboard item | Comes from | Needs the model |
|---|---|---|
| Daily OFF time (tile, trend, first-week baseline, `off_time_increase` flag) | Watch | Yes |
| Daily dyskinesia (tile, trend, `dyskinesia_time` flag) | Watch | Yes |
| Tremor estimate with an uncertainty band (timeline) | Watch | Yes |
| Dyskinesia likelihood (timeline) | Watch | Yes |
| Dose response: tremor from before a dose to hours after | Watch, lined up with the dose log | Yes |
| Wear time (tile, trend, `low_wear_time` flag) | Watch: worn or not | No |
| Last sync (`not_synced` flag) | Phone uploads | No |
| Dose adherence, dose markers (`doses_not_logged` flag) | App dose log against the prescription | No |
| Check-ins, fall reported (`fall_reported` flag) | App | No |

### Outputs the dashboard expects

From Careportal's monitoring types (`src/features/monitoring/model/monitoring.ts`) and the proposed contract (`openapi/proposed.json`):

| Output | Scale | How often |
|---|---|---|
| Tremor estimate | MDS-UPDRS 0 to 4 | Every 10-minute interval |
| Tremor uncertainty, low and high | 0 to 4 | Every 10-minute interval |
| Dyskinesia likelihood | 0 to 1 | Every 10-minute interval |
| OFF or not | Counted against an OFF threshold | Every 10-minute interval, summed into hours |

Each score should also carry the model version that made it, for the registry, rollback and drift.

**Today:** the phone model sends one OFF risk (0 to 1) per hour. Backend's monitoring endpoints fill the tremor line with OFF risk × 4, with no band, and dyskinesia as 0. The dashboard shows real data only once the model produces the outputs above.

## 2. Definitions

| Term | Meaning |
|---|---|
| **OFF time** | Time when the Parkinson's medicine isn't working and symptoms return: mainly slowness and stiffness (akinesia, bradykinesia), often tremor, sometimes freezing. Daily OFF time is the hours per day spent OFF. |
| **ON** | Medicine working; moving well. |
| **Dyskinesia** | Extra, involuntary movement, usually from too much medicine. |
| **Wearing off** | Each dose lasting less long as the disease progresses, so OFF returns before the next dose. |

**How the sources count OFF differently:**

- **COPS diary:** OFF is the akinesia entries (slight, discomforting or severe akinesia). The data pipeline maps them to `kinesia_class = off`.
- **Careportal:** OFF time is the 10-minute intervals where the **tremor** estimate is above the OFF threshold (2 on the 0 to 4 scale), summed into hours (`src/features/monitoring/charts/timeline.ts`, `offHours`).

These don't match: the labels define OFF by akinesia, the dashboard by tremor. See [Decisions](#decisions).

## 3. Data

COPS ([osf.io/5xvwn](https://osf.io/5xvwn/)): 66 people with Parkinson's, up to 7 days each, at home or in hospital. 46 have DBS.

| Data | How often | Notes |
|---|---|---|
| Wrist accelerometer, X/Y/Z in g | 100 Hz | Both wrists; the data pipeline uses one (`features.wrist`) |
| Diary: kinesia (akinesia, good, dyskinesia, asleep), tremor 0 to 2, freezing, falls, watch worn | Once per waking hour | The only labels. Self-reported. |
| Diary: medication taken, which medicines, dose in mg | Per hour | A tick per hour, not a time to the minute |
| MDS-UPDRS III totals, OFF and ON | Once per person | |
| Demographics: age, sex, handedness, subtype, more affected side, Hoehn and Yahr, years since diagnosis, DBS | Once per person | |

**The labels are hourly.** Anything the model scores more often than once an hour can only be trained and checked against the hour it falls in.

## 4. Inputs

What's available. Which of these the model uses is in [Decisions: Inputs](#inputs) and [Dose times](#dose-times).

### Movement

The data pipeline computes 7 features per hour from the accelerometer (`Data_Pipeline/scripts/features.py`): the signal is band-passed at 0.1 to 3 Hz (movement) and 4 to 8 Hz (tremor), averaged into 1-second epochs, then summarised.

| Feature | What it measures |
|---|---|
| `move_mean` | Average movement |
| `move_std` | How much movement varies |
| `move_p95` | The strongest movement (95th percentile) |
| `still_fraction` | Share of seconds below 0.01 g of movement |
| `tremor_mean` | Average tremor-band energy |
| `tremor_p95` | The strongest tremor moments |
| `tremor_ratio` | Tremor divided by movement |

The same maths runs in the pipeline, the Android app and Backend, tested against golden values. The raw accelerometer is also available to any input built from it.

### Medication

The app logs each dose with its time (`POST /v1/me/doses`), and Backend has the prescription schedule, so medication values are available live as well as in training.

**`hours_since_dose`, as the data pipeline builds it today** (`Data_Pipeline/scripts/diary.py`, `add_hours_since_dose`):

- From the diary's "medication taken" tick: any medicine counts as a dose.
- In whole hours: 0 in the dose hour, then 1, 2, and so on until the next tick.
- Empty before the patient's first logged dose.
- Runs on across nights.

**Differences between training and live:**

- Training values are whole hours (the diary's limit); the app's logs are to the minute.
- A forgotten or late log looks like a long gap live.
- DBS patients depend less on pill timing.

#### Dose equivalence (LEDD)

COPS patients take many Parkinson's drugs, not only levodopa. The levodopa equivalent daily dose (LEDD) converts each drug to the mg of levodopa it's worth, so doses of different drugs can be added up.

- Tomlinson et al., 2010, _Movement Disorders_: the original conversion table.
- Jost et al., 2023, _Movement Disorders_: the updated table, which adds newer drugs such as opicapone and safinamide.

Drugs in the 5-participant sample (`hourly_all.csv`, names as COPS writes them). The full 66 may add more.

| Drug | Class |
|---|---|
| Levodopa / Benserazid (plain, RE, ER, FA) | Levodopa: immediate, extended and fast-acting forms |
| Levodopa / Carbidopa, with or without Entacapone | Levodopa, sometimes with a COMT inhibitor |
| Amantadine | NMDA antagonist |
| Ropinirole, Ropinirole ER | Dopamine agonist |
| Pramipexole RE | Dopamine agonist |
| Piribedil ER | Dopamine agonist |
| Opicapone | COMT inhibitor |
| Safinamide | MAO-B inhibitor |
| Rasagiline | MAO-B inhibitor |

**Notes on using LEDD:**

- It's defined for daily totals. Converting single doses is a derived use.
- Drugs act on different clocks: immediate-release levodopa peaks within an hour and wears off within hours; extended-release forms spread over many hours; MAO-B inhibitors are taken once a day and act all day. Time since a dose means different things by drug.
- COMT inhibitors (entacapone, opicapone) only extend levodopa; LEDD adds them to the levodopa dose.
- COPS drug names need cleaning before a lookup (spacing, and suffixes such as RE, ER, FA).
- The conversion factors go in the data pipeline's config, taken from the paper, with the citation. Not from memory.

## 5. Model structure

The 3 outputs can come from one model with an output per symptom, or from separate models.

| | One model, 3 outputs | Separate models |
|---|---|---|
| On the phone | One file, one run per window | One file and one run each |
| Training | One run; the outputs share what they learn | One run each, tuned separately |
| Changing one output | Retrains all of them | Swaps one alone |
| Registry and rollback | One version | Versions kept in step |

## Decisions

Terms used below:

- **Window:** 10 seconds of accelerometer data, the unit the encoder reads.
- **Block:** 10 minutes, so 60 windows. The unit the history part steps through.
- **Hour:** 6 blocks, so 360 windows. The unit the diary labels.

### Outputs and how often

_Owner: Sam Selvaraj · Decided: 2026-11-11_

**Decision (2026-11-11):**

- Two horizons: **now** (the last 60 minutes) and **next hour** (the 60 minutes after).
- At each horizon, the **state** as 4 probabilities that add up to 1: P(asleep), P(OFF), P(ON), P(dyskinetic). They come from the diary's kinesia entries: Sleep; akinesia −3 to −1; good 0; dyskinesia +1 to +3.
- At each horizon, **tremor** as P(tremor ≥ 1) and P(tremor ≥ 2), from the diary's 0 to 2 tremor score. Tremor is labelled on waking hours only.
- **Expected tremor** for the timeline is P(≥ 1) + P(≥ 2), on the diary's 0 to 2 scale.
- Scores are made per block, with a per-window timeline inside each block. Only the hourly scores are trained and checked against the diary; the window timeline shows roughly when something happened, not exactly.
- Daily OFF time is the number of hours whose P(OFF) is above the OFF threshold.
- Every score carries the version of the model that made it.

**Why:** The diary is the only ground truth, and it labels whole hours with these answers, so the model predicts what the diary can check. One state with 4 classes keeps asleep, OFF, ON and dyskinetic mutually exclusive, as they are in the diary. Tremor is separate and ordinal, because it can happen in any waking state. The next-hour horizon is the warning of an OFF period coming.

**Changes elsewhere:**

- Careportal's tremor line moves from MDS-UPDRS 0 to 4 to the diary's 0 to 2.
- `offHours` counts from P(OFF), not from tremor.
- The dyskinesia likelihood is P(dyskinetic).
- Backend's monitoring endpoints stop deriving tremor from OFF risk.

### How OFF is defined and counted

_Owner: Sam Selvaraj · Decided: 2026-11-10_

Akinesia (the labels) or tremor above a threshold (the dashboard today). Section 2.

**Decision (2026-11-10):** OFF is the diary's akinesia entries (kinesia score −3 to −1, `kinesia_class = off`), as in clinical practice. Tremor is a separate target, not the definition of OFF. Hours with no diary answer (asleep, no data) have no OFF label, not a 0. Defined in `Model_Pipeline/scripts/core/targets.py`.
**Why:** Clinicians count OFF time with patient diaries (the Hauser diary, MDS-UPDRS Part IV), where OFF is the medicine wearing off, mainly slowness and stiffness. Many patients have little or no tremor, so a tremor threshold would miss their OFF time and count tremor while ON as OFF. Careportal's `offHours` (tremor above 2) will need to count from the OFF output instead.

**Update (2026-11-11):** Asleep is now one of the 4 states (see Outputs), so diary "Sleep" hours are labelled asleep instead of having no label. Hours with no diary answer still have no label. The definition of OFF is unchanged. `core/targets.py` changes to match.

### Inputs

_Owner: Sam Selvaraj · Decided: 2026-11-11_

Movement features, raw accelerometer, medication values, LEDD. Section 4.

**Decision (2026-11-11):**

- **Movement:** the raw accelerometer, converted to the model rate (30 Hz by default) and cut into windows of 3 axes × 300 samples, with each axis's mean removed.
- **The 7 hourly features aren't model inputs.** They stay for the baselines and for drift monitoring.
- **Context, per block:**
  - minutes since the last dose, and that dose's source flag (see Dose times);
  - last-dose LEDD, and today's LEDD so far;
  - time since waking;
  - worn or not;
  - a check-in if there is one, with a flag saying whether there is.
- **Per patient:** watch on the more affected side or not; DBS or not.
- **Not inputs:**
  - diary answers, which are the targets;
  - age, sex and disease stage, which are used for bias checks only.

**Why:**

- The 7 features summarise an hour in 7 hand-made numbers. Raw windows let the encoder learn its own features and start from harnet10, which was pretrained on 10-second windows at 30 Hz.
- Doses, waking and side help interpret movement, but don't describe it, so they enter after the encoder.
- Demographics are left out so the model can't take shortcuts that don't hold from one patient to the next.

### Dose times

_Owner: Sam Selvaraj · Decided: 2026-11-11_

**Decision (2026-11-11):**

- **COPS doses go at mid-hour (±30 min),** because the diary only ticks the hour. This replaces the whole-hour `hours_since_dose` (Section 4).
- **Live, the app sets a source flag on each dose:**
  - on time: entered close to the dose time;
  - logged late: entered long after the dose time;
  - none logged.
- **Training adds errors to the dose inputs,** drawn fresh each epoch:
  - times shifted at random, up to ±60 min (tuned);
  - doses dropped;
  - rarely, an extra dose added.

  The flag is set to match the error added.
- **The app asks at each scheduled dose time,** for example "Did you take your 8:00 dose?" (AndroidApp, Ayush).
- **Every run reports accuracy** with doses shifted and with doses hidden.

**Why:** Live dose times are only as exact as the patient's logging: late, forgotten, or now and then logged but not taken. Training on imprecise times keeps the model from depending on exact times, so the movement stays the main evidence.

### Preparing the signal

_Owner: Sam Selvaraj, with Dhruv for the data pipeline · Decided: 2026-11-11_

**Decision (2026-11-11):**

- **One model rate.** Every device is converted to it where data enters: measure the true rate from timestamps, low-pass filter, then resample.
- **Watch-like training copies.** Training also uses copies of each window passed through 24 Hz (100 → 24 → 30 Hz), so they look like the CMF watch.
- **Masking.** Windows when the watch wasn't worn are masked, never scored.
- **One preprocessing spec,** versioned, with golden test values. The data pipeline and the app run the same steps.

**Why:** The model never sees a device's native rate, so supporting a new watch needs only its converter. The CMF watch samples at about 22 to 24 Hz, so it carries nothing above about 12 Hz, which COPS at 30 Hz does.

### Architecture

_Owner: Sam Selvaraj · Decided: 2026-11-11_

**Decision (2026-11-11):**

- **Window encoder:** reads one window, movement only. The starting point is the hybrid from the architecture PDF:
  - frozen harnet10;
  - a trained 1D CNN on the raw signal;
  - a trained 2D CNN on the spectrogram (0 to 15 Hz).

  They're joined into a 256-number window embedding, plus window scores. Nothing else in the design depends on which encoder is used.
- **Block summary:** a block's 60 window embeddings are pooled with gated attention into one summary. Unworn windows get no weight.
- **History part:**
  - reads one step per block, in order. A step is the block summary joined with that block's context, with side and DBS added to every step;
  - variants to test: average, GRU, attention;
  - history of 2, 4 or 8 hours (12, 24 or 48 blocks).
- **Two heads, now and next hour:** the same inputs and architecture, with the label shifted by an hour. Sharing one encoder or training two copies is a config switch.
- **The model's own past outputs are not an input.**

**Why:**

- **Medication enters after the encoder,** so the encoder's reading of the body doesn't depend on dose timing.
- **Blocks rather than hours or windows.** Hourly steps would round "time since dose" to the hour. Steps per window would mean 1,440 steps for 4 hours, trained from one label per 360 steps. Blocks keep dose timing to 10 minutes while keeping sequences short enough to train from hourly labels with 66 patients.
- **Past outputs aren't fed back** because, live, there are no labels to correct them, so a wrong output would feed the next. The history part already sees the block summaries those outputs would come from.

### Training

_Owner: Sam Selvaraj · Decided: 2026-11-11_

**Decision (2026-11-11):**

- **Phase 1, the encoder:**
  - train the encoder and a simple head on single hours: 360 windows in, the hour's diary label out. The attention pooling combines the window scores, so the hourly label trains the window scorer (multiple-instance learning);
  - then freeze the encoder and store one summary per block.
- **Phase 2, the history part and the heads:**
  - examples slide along each patient's blocks and end on each hour. Each example has two labels: that hour, and the next;
  - no-data blocks stay in the history but are never targets;
  - check-ins are simulated by revealing a few random past diary hours;
  - dose-time errors are added as in Dose times.
- **Splits:**
  - folds are by patient;
  - the test patients are fixed and never used for training, tuning or any choice;
  - 3 seeds.
- **Later variant:** fine-tune the encoder and history part together, end to end, if the experiments show the frozen encoder holds the history part back.

**Why:** All the heavy 30 Hz work happens once, in phase 1. Phase 2 is small and fast, so many history lengths, variants and seeds can be run, which is what the hyperparameter sensitivity analysis needs.

### On the phone

_Owner: Sam Selvaraj, with Ayush for the AndroidApp · Decided: 2026-11-11_

**Decision (2026-11-11):**

- **On the phone, not the server.** The model runs on the phone, quantised to int8, each time a block completes.
- **Raw movement never leaves the phone.** The uploads are the scores, the timeline, the 7 features and an embedding summary.
- **Runtime:** chosen by a conversion spike that tries ExecuTorch, ONNX Runtime and LiteRT. The winner is the one that converts with matching outputs and the least rewriting.
- **One `ModelRunner` interface** in the app, with a runner per format.
- **A manifest with each model** names its format, its input and output shapes, and a reference input with expected outputs. The app checks it before scoring a patient.
- **At night,** Health Connect sleep sessions are the main signal. The model's asleep output covers the time before they sync, and acts as a cross-check.

**Why:**

- **Privacy:** raw movement is the most sensitive data.
- **Offline:** scoring works without a connection.
- **Cost:** only new blocks are encoded.

### One model or separate models

_Owner: Sam Selvaraj · Decided: 2026-11-11_

Section 5.

**Decision (2026-11-11):** One architecture, with the now head and the next-hour head each giving the state and tremor. Whether the two heads share one encoder or use two trained copies is decided by experiment stage 3. The tiebreak is still open.
**Why:** The state and tremor are read from the same movement, so they share what the encoder learns. The two horizons differ only in their label, so they can share the encoder too, unless the experiments show otherwise.

### Release gate and retraining

_Owner: Sam Selvaraj · Decided: 2026-11-11_

**Decision (2026-11-11):** A new model may replace the current one only if:

- **it improves** OFF-now fold AUC across the patient folds;
- **it improves or holds** OFF for non-DBS patients;
- **it doesn't get worse** for tremor, dyskinesia, asleep or next-hour OFF.

Every run also reports these, without gating:

- within-patient tracking;
- pooled AUC with a 95% interval;
- the spread across 3 seeds;
- OFF in the hours shortly after a dose;
- accuracy with doses hidden and with doses shifted;
- how often a still OFF hour is scored asleep;
- shuffled labels, which must fall to chance;
- bias by sex, age group, DBS and stage.

**Retraining is triggered separately,** by monitoring live data:

- input drift;
- output drift: the share of hours scored OFF, asleep and so on;
- agreement with check-ins.

Drift can start a retrain, but only the gate changes what patients get.

**Why:** OFF is what clinicians act on. Most real users don't have DBS, and only about 20 COPS patients don't, so non-DBS OFF is checked on its own.

### Experiment order

_Owner: Sam Selvaraj · Decided: 2026-11-11_

**Decision (2026-11-11):** Every stage uses the same folds, seeds and gate. Each variant is its own MLflow run, with nested fold runs.

| Stage | Runs | Answers |
|---|---|---|
| 0. Baselines | training average · logistic regression on the 7 features · shuffled labels | the floor, and that the pipeline works end to end |
| 1. Encoder | phase 1 on single hours (harnet10 alone as the bar to beat) | how well movement alone reads the state |
| 2. History | 2, 4 and 8 hours · average vs GRU vs attention | how much the past helps |
| 3. Heads | shared encoder vs two trained copies | one model or two for the two horizons |
| 4. Medication | with vs without dose inputs · dose-time error sizes · with vs without the source flag | whether doses help, and how much timing error the model tolerates |
| 5. Watch realism | with vs without the 24 Hz copies | whether the model survives the watch's lower rate |

### LEDD conversion table

_Owner: Sam Selvaraj · Decide by: TBD_

Which paper's factors, and how drug names in COPS map to them.

**Decision:** _TBD_
**Why:** —

### Still open

- **Model rate:** 30 Hz by default, which keeps harnet10's pretraining; or test 24 Hz as well.
- **How often to score live:** every 10 minutes, one block, which fits the design; or once an hour.
- **A live state between block ends:** pool the windows so far into a partial block, or smooth the window timeline (the architecture PDF's causal 15 minutes), or both.
- **History variants beyond the sliding window:** a GRU that carries its memory forward block to block; a memory per window for the last 10 to 30 minutes; label-free pretraining that predicts the next block.
- **Dose-time error sizes:** ±60 min proposed; how often to drop or add a dose. Tuned in stage 4.
- **Tiebreak** if shared and separate heads score the same.
- **OFF threshold** for counting daily OFF hours.
- **Runtime,** until the conversion spike runs.
- **Tremor uncertainty band** on Careportal's timeline, from the predicted 0, 1, 2 probabilities.
