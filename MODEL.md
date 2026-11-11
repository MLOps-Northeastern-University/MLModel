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

- **Window:** 10 seconds of accelerometer data. The unit the model classifies.
- **Token:** one window as the history attention sees it: the window's embedding, the patient fields (side, DBS) and its event fields (a dose's LEDD and source flag, a check-in's answers).
- **Diary hour:** the unit the COPS diary labels, so 360 windows.

The design was drafted on 2026-11-11 around 10-minute blocks and revised the same day to the per-window model below, before any code depended on it. The entries below replace that draft.

### Outputs and how often

_Owner: Sam Selvaraj · Decided: 2026-11-11_

**Decision (2026-11-11):**

- Two horizons: **now** (the state of the current window) and **in 1 hour**.
- At each horizon, the **state** as 8 probabilities that add up to 1, one softmax: the diary's kinesia score −3 to +3 (severe akinesia … good … severe dyskinesia) and asleep.
- At each horizon, **tremor** as 3 probabilities, one softmax over the diary's tremor score: 0 none, 1 weak or slight, 2 strong or severe.
- **Everything else is worked out from those probabilities:**
  - P(OFF) is the sum of scores −3 to −1, P(ON) is score 0, and P(dyskinetic) is the sum of +1 to +3;
  - the expected kinesia score and expected tremor (Σ score × probability) are the timeline's lines;
  - P(any tremor) is P(1) + P(2);
  - daily OFF and dyskinesia time can be split by severity.
- **Sleep hours are left out of the tremor target.** The diary does record tremor in most sleep hours (almost always 0), but tremor while asleep is trivially none and the asleep state already covers it, so keeping them would inflate the tremor metrics without teaching anything.
- **A state every 10-second window.** Only the hourly scores (the window outputs pooled per diary hour) are trained and checked against the diary, so the window states show roughly when something happened, not exactly.
- Daily OFF time is the number of hours whose pooled P(OFF) is above the OFF threshold.
- Every score carries the version of the model that made it.

**Why:** The diary is the only ground truth, and it labels whole hours with these answers, so the model predicts what the diary can check. One state keeps asleep, OFF, ON and dyskinetic mutually exclusive, as they are in the diary. Predicting the score instead of the 4 broad states keeps the severity: in the sample, most waking hours are akinesia or dyskinesia of some severity, and only 23 of 127 are good, so an hour of slight akinesia and one of severe akinesia shouldn't count the same. The broad states can always be worked out from the score, but not the other way round. Tremor is a separate target, because it can happen in any waking state. Both are classification over ordered scores, so the training loss also counts how far a prediction is from the true score (see Training). The 1-hour horizon is the warning of an OFF period coming. Per-window states follow from the encoder: harnet10 is pretrained on 10-second windows.

**Changes elsewhere:**

- Careportal's tremor line moves from MDS-UPDRS 0 to 4 to the diary's 0 to 2, and the timeline shows per-window states.
- `offHours` counts from P(OFF), not from tremor.
- The dyskinesia likelihood is P(dyskinetic).
- Backend's monitoring endpoints stop deriving tremor from OFF risk.

### How OFF is defined and counted

_Owner: Sam Selvaraj · Decided: 2026-11-10_

Akinesia (the labels) or tremor above a threshold (the dashboard today). Section 2.

**Decision (2026-11-10):** OFF is the diary's akinesia entries (kinesia score −3 to −1, `kinesia_class = off`), as in clinical practice. Tremor is a separate target, not the definition of OFF. Hours with no diary answer (asleep, no data) have no OFF label, not a 0. Defined in `Model_Pipeline/scripts/core/targets.py`.
**Why:** Clinicians count OFF time with patient diaries (the Hauser diary, MDS-UPDRS Part IV), where OFF is the medicine wearing off, mainly slowness and stiffness. Many patients have little or no tremor, so a tremor threshold would miss their OFF time and count tremor while ON as OFF. Careportal's `offHours` (tremor above 2) will need to count from the OFF output instead.

**Update (2026-11-11):** Asleep is now one of the 4 states (see Outputs), so diary "Sleep" hours are labelled asleep instead of having no label. Hours with no diary answer still have no label. The definition of OFF is unchanged. `core/targets.py` changes to match.

**Update (2026-11-11, later):** The state is now the kinesia score −3 to +3 plus asleep (see Outputs). OFF is still akinesia: the sum of the −3 to −1 classes.

### Inputs

_Owner: Sam Selvaraj · Decided: 2026-11-11_

Movement features, raw accelerometer, medication values, LEDD. Section 4.

**Decision (2026-11-11):**

- **Movement:** the raw accelerometer, converted to the model rate (30 Hz by default) and cut into windows of 3 axes × 300 samples, with each axis's mean removed.
- **The 7 hourly features aren't model inputs.** They stay for the baselines and for drift monitoring.
- **Patient fields, on every window's token:** watch on the more affected side or not; DBS or not.
- **Event fields, filled only on the window where something happened, 0 everywhere else:**
  - a dose taken in that window: its LEDD (doses in the same window add up) and its source flag;
  - a check-in made in that window: the patient's two answers.
- **Nothing is carried forward.** No time since the dose, no last dose, no daily total. Attention reads every dose and check-in in the 8-hour history, with their timing from the time-based positions.
- **Sleep windows are scored and added to the history,** like any other window; the model learns asleep from them.
- **Unworn windows** carry no movement, so they aren't scored and aren't added to the history. A dose or check-in during them goes on the next window that is added.
- **Not inputs:**
  - diary answers, which are the targets (check-ins are the patient's own live answers, not diary labels);
  - age, sex and disease stage, which are used for bias checks only.

**Why:**

- The 7 features summarise an hour in 7 hand-made numbers. Raw windows let the encoder learn its own features and start from harnet10, which was pretrained on 10-second windows at 30 Hz.
- Doses and side help interpret movement but don't describe it, so they join after the encoder.
- **Doses as LEDD only, read from the history (2026-11-11):**
  - LEDD already puts every drug on one scale, so the drug class is dropped;
  - with 8 hours of history, a typical day's doses are all in view together, so attention learns both each dose's own effect and how overlapping doses add up;
  - carried-forward values (time since dose, last dose, today's LEDD) were dropped: they handled one dose at a time, today's total gave away the time of day and reset at midnight, and the history already holds the same information.

  If the experiments show the model can't use dose timing from the positions alone, add back one number: time since the last dose.
- **Source flag:** on time, logged late, or none logged (see Dose times), so the model can trust a guessed dose time less.
- Demographics are left out so the model can't take shortcuts that don't hold from one patient to the next.
- **Sleep stays in the history (2026-11-11):**
  - it costs nothing extra, because the history is capped at 8 hours either way;
  - leaving it out would need a "this is sleep" rule, and none works the same in training and live: in training the only marker is the diary's Sleep hours, which are the labels, while live it would be Health Connect or the model's own P(asleep);
  - attention can learn to ignore it, and sometimes it matters: a restless night, an early waking, or the early-morning OFF before the first dose.

  The cost: in the morning the history is mostly sleep, so the previous evening's movement and doses have dropped out.
- **No time since waking or time of day (2026-11-11):** with sleep in the history, the model can see when the patient woke and roughly where in the day it is. A clock input could also teach shortcuts ("8 am means OFF") that fail for patients with other routines. Add time of day only if the errors show a time-of-day pattern, such as missed early-morning OFF.

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

### Check-ins

_Owner: Sam Selvaraj, with Ayush for the AndroidApp · Decided: 2026-11-11_

**Decision (2026-11-11):**

- **Both diary questions,** with the diary's own answer options: how the patient is moving (the 7 kinesia answers: akinesia and dyskinesia each slight, discomforting or severe, or good) and how much tremor (none, weak, slight, strong or severe, scored 0 to 2).
- **3 to 5 a day,** mostly at random times, some just before a dose and about an hour after. Each records why it was asked.
- **In the model,** the answers go in the event fields of the window where the patient answered, and attention reads them from the history; they aren't carried forward. The model's own past outputs are never fed back; check-ins are the only past labels it sees.
- **In training,** check-ins are simulated at the live rate: a few random diary hours per day are revealed on one random window of the hour.
- **Hours with a check-in are scored separately,** because they're easy. The gate is judged on hours without one.
- **When the model and a check-in disagree,** the disagreement feeds monitoring (agreement with check-ins), and the app may show the patient's own answer for that moment.

**Why:** The answers line up with what the model is trained on only if they use the diary's scale. Simulating them at the live rate teaches the model to use them without waiting for them. Check-ins are also the first labels from real MedNova users, for retraining later.

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

- **Window encoder:** reads one window, movement only, and gives 256 numbers. The starting point is the hybrid from the architecture PDF:
  - frozen harnet10: general wrist movement, from about 700,000 days of UK Biobank data;
  - a trained 1D CNN on the raw signal: Parkinson's-specific movement shapes;
  - a trained 2D CNN on the spectrogram (0 to 15 Hz): frequency, where tremor shows.

  Each branch is a config switch. Experiment stage 1 decides which to keep, starting from harnet10 alone.
- **Token:** the embedding projected from 256 to 64 numbers (tunable), joined with the patient fields and the window's event fields. One token per window.
- **History attention:**
  - a small causal transformer (2 layers, 2 to 4 heads, about 64 dimensions) over the tokens of the last 8 hours;
  - the new token is the query, and the cached tokens are the keys and values;
  - each token's keys and values are computed once and kept in a KV cache;
  - positions come from each token's real timestamp (time-based RoPE or ALiBi), because the cache skips unworn time, so neighbouring tokens can be hours apart.
- **Two heads:**
  - **now:** the state of this window;
  - **in 1 hour:** the state an hour from now.

  Both read the same attention output; only the label differs. The forecast is a direct head, not a step-by-step rollout.
- **Baseline for experiment stage 2:** the same model with the attention replaced by fixed summaries of the past (movement over the last 5, 30 and 60 minutes, and since the last dose), each read by a small MLP.

**Why:**

- **Per window:** harnet10 is built for 10-second windows, and per-window states are what the app shows.
- **Attention over hours of past:** a dose takes 20 to 40 minutes to work, peaks around an hour and wears off over 2 to 4 hours. Judging "this dose is wearing off early" needs the baseline before the dose and the path since, and attention learns what in the past matters instead of us picking spans.
- **KV cache:** each new window costs one attention step, about 0.8 M multiply-adds over 2,880 cached tokens (8 hours), with a cache of about 3 MB. The past is never recomputed.
- **256 → 64:** a smaller attention model overfits less with 66 patients, and the cache stays small.
- **Direct forecast:** rolling forward window by window would take 360 steps to reach an hour, with errors compounding.
- **No 10-minute blocks:** with attention and a KV cache, compressing windows into blocks isn't needed to keep the past cheap.

### Training

_Owner: Sam Selvaraj · Decided: 2026-11-11_

**Decision (2026-11-11):** multiple-instance learning (MIL), with each diary hour as a bag of 360 windows.

1. **Run the whole model on a patient-day at once.** That's about 8,640 windows, with a causal mask limited to the last 8 hours. Every window gets its now and in-1-hour outputs with its full past, the same as live with the KV cache.
2. **Pool per diary hour.** Gated attention pools the hour's 360 now-outputs into one hour prediction. Unworn windows get no weight.
3. **Compare with the diary.** The hour prediction is compared with the hour's label, and the in-1-hour outputs are pooled the same way and compared with the next hour's label. No-data hours are never targets. The loss for each target is cross-entropy + λ × expected distance, where the expected distance is Σ P(score) × |score − true score|, so a near miss costs less than a far one. Cross-entropy keeps the probabilities calibrated; the distance alone would push them onto one middle score. λ is tuned. Asleep has no score, so the distance term covers waking hours only.
4. **Backpropagate** through the pooling, heads, attention, token projection and the trained encoder branches. harnet10 stays frozen.

Also:

- **Caching:** harnet10's outputs are computed once and cached.
- **Baseline:** giving every window its hour's label, to show what MIL adds.
- **Noise and simulated check-ins:** the dose-time errors and simulated check-ins above.
- **Splits:**
  - folds are by patient;
  - the test patients are fixed and never used for training, tuning or any choice;
  - 3 seeds.
- **Later variant:** fine-tune harnet10's last layers, if the experiments show the frozen encoder limits the model.

**Why:** The diary labels hours, not windows. MIL lets the model decide which windows explain an hour's label. That matters for tremor and dyskinesia, which come and go within an hour, while OFF usually lasts all of it. The per-window outputs become meaningful because the hour prediction is built only from them. Running whole days with a causal mask gives every window its full past in one pass.

### On the phone

_Owner: Sam Selvaraj, with Ayush for the AndroidApp · Decided: 2026-11-11_

**Decision (2026-11-11):**

- **On the phone, not the server.** The model runs on the phone, quantised to int8, over each new window, in batches whenever the app wakes.
- **Cost per window:**
  - the encoder, about 15 ms on CPU (about 5 s of CPU per hour of data);
  - one attention step against the KV cache (about 3 MB for 8 hours);
  - the heads.
- **The KV cache comes back after a restart.** If Android kills the app, the cache is rebuilt from the stored window embeddings by rerunning only the attention layers.
- **Raw movement never leaves the phone.** The uploads are the window states, the hourly scores and the 7 features.
- **Runtime:** chosen by a conversion spike that tries ExecuTorch, ONNX Runtime and LiteRT. The spike tests the KV cache specifically, because a stateful model is the hard part of the export.
- **One `ModelRunner` interface** in the app, with a runner per format.
- **A manifest with each model** names its format, its input and output shapes, and a reference input with expected outputs. The app checks it before scoring a patient.
- **At night,** the model scores every worn window, asleep included. Health Connect sleep sessions are a cross-check on the model's asleep output.

**Why:**

- **Privacy:** raw movement is the most sensitive data.
- **Offline:** scoring works without a connection.
- **Cost:** only new windows are encoded, and the past is never recomputed.

### One model or separate models

_Owner: Sam Selvaraj · Decided: 2026-11-11_

Section 5.

**Decision (2026-11-11):** One model, with the now head and the in-1-hour head each giving the state and tremor. Whether the in-1-hour head shares the trunk or uses a separate trained copy is decided by experiment stage 3. The tiebreak is still open.
**Why:** The state and tremor are read from the same movement, so they share what the encoder learns. The two horizons differ only in their label, so they can share the trunk too, unless the experiments show otherwise.

### Release gate and retraining

_Owner: Sam Selvaraj · Decided: 2026-11-11_

**Decision (2026-11-11):** A new model may replace the current one only if, judged on hours without a check-in:

- **it improves** OFF-now fold AUC across the patient folds;
- **it improves or holds** OFF for non-DBS patients;
- **it doesn't get worse** for tremor, dyskinesia, asleep or in-1-hour OFF.

Every run also reports these, without gating:

- within-patient tracking;
- pooled AUC with a 95% interval;
- the spread across 3 seeds;
- OFF in the hours shortly after a dose;
- accuracy with doses hidden and with doses shifted;
- how often a still OFF hour is scored asleep;
- hours with a check-in, separately;
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
| 1. Encoder | harnet10 alone · + spectrogram CNN · + raw-signal CNN · all three · MIL vs the hour's label on every window | which branches earn their place, and whether MIL helps |
| 2. History | history attention over 8 h, then 4 h · against the fixed-summary baseline | how much the past helps, and whether a shorter history does as well |
| 3. Heads | shared trunk vs a separate copy for the in-1-hour head | one model or two for the two horizons |
| 4. Medication | with vs without the dose LEDD · dose-time error sizes · with vs without the source flag · with vs without an explicit time since the last dose | whether doses help, and how much timing error the model tolerates |
| 5. Watch realism | with vs without the 24 Hz copies | whether the model survives the watch's lower rate |

### LEDD conversion table

_Owner: Sam Selvaraj · Decide by: TBD_

Which paper's factors, and how drug names in COPS map to them.

**Decision:** _TBD_
**Why:** —

### Still open

- **Model rate:** 30 Hz by default, which keeps harnet10's pretraining; or test 24 Hz as well.
- **Attention size:** 2 layers × 64 dimensions proposed. The history is 8 h; stage 2 checks whether 4 h does as well.
- **Time encoding:** time-based RoPE or ALiBi.
- **Token size:** 64 proposed; 32, 128 or the full 256 to try.
- **Dose-time error sizes:** ±60 min proposed; how often to drop or add a dose. Tuned in stage 4.
- **Tiebreak** if a shared trunk and a separate in-1-hour copy score the same.
- **OFF threshold** for counting daily OFF hours.
- **Runtime,** including the KV cache, until the conversion spike runs.
- **Tremor uncertainty band** on Careportal's timeline, from the predicted 0, 1, 2 probabilities.
