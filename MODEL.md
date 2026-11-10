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

### Outputs and how often

_Owner: Sam Selvaraj · Decide by: TBD_

**Decision:** _TBD_
**Why:** —

### How OFF is defined and counted

_Owner: Sam Selvaraj · Decide by: TBD_

Akinesia (the labels) or tremor above a threshold (the dashboard today). Section 2.

**Decision (2026-11-10):** OFF is the diary's akinesia entries (kinesia score −3 to −1, `kinesia_class = off`), as in clinical practice. Tremor is a separate target, not the definition of OFF. Hours with no diary answer (asleep, no data) have no OFF label, not a 0. Defined in `Model_Pipeline/scripts/core/targets.py`.
**Why:** Clinicians count OFF time with patient diaries (the Hauser diary, MDS-UPDRS Part IV), where OFF is the medicine wearing off, mainly slowness and stiffness. Many patients have little or no tremor, so a tremor threshold would miss their OFF time and count tremor while ON as OFF. Careportal's `offHours` (tremor above 2) will need to count from the OFF output instead.

### Inputs

_Owner: Sam Selvaraj · Decide by: TBD_

Movement features, raw accelerometer, medication values, LEDD. Section 4.

**Decision:** _TBD_
**Why:** —

### LEDD conversion table

_Owner: Sam Selvaraj · Decide by: TBD_

Which paper's factors, and how drug names in COPS map to them.

**Decision:** _TBD_
**Why:** —

### One model or separate models

_Owner: Sam Selvaraj · Decide by: TBD_

Section 5.

**Decision:** _TBD_
**Why:** —
