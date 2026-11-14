# Data Pipeline

Turns the COPS Parkinson's dataset ([osf.io/5xvwn](https://osf.io/5xvwn/)) into the inputs the model trains on: 10-second wrist-movement windows, every diary hour with its answers, every dose, and one row per patient. These are the input groups of `Model_Pipeline/scripts/core/data.py`; the model design is in [MODEL.md](../MODEL.md).

## Dataset

**COPS**: 66 people with Parkinson's, up to 7 days each, one zip per participant (47.9 GB; the 5 smallest, used by `sample` mode, are 1.1 GB). Check the OSF page for its licence, and cite it as the page asks.

- **Wrist accelerometers** (GENEActiv) on both wrists: one CSV per wrist per hour, 100 samples a second: `X`, `Y`, `Z` (g), `Photo`, `Temp`.
- **An hourly symptom diary:** kinesia (−3 to +3, or Sleep), tremor (0 to 2), medication and dose, and whether the watch was worn. The only labels.
- **`Demographics.csv`:** age, sex, PD subtype and more affected side, Hoehn & Yahr stage, DBS.

The zips also hold UPDRS exams; the pipeline doesn't use them. Participant IDs are the dataset's pseudonyms (`COPS-7`); files that list participants are versioned by DVC, not committed.

## Quick start (only Docker needed)

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and give it at least 4 GB of memory. Then, from this folder:

```bash
mkdir -p logs/airflow && echo "AIRFLOW_UID=$(id -u)" > .env    # Linux only, once
docker compose run --rm tests       # the tests, in the Airflow image (the first run builds it)
docker compose run --rm pipeline    # the pipeline on the 5 smallest participants (~1.1 GB download)
docker compose up airflow-init      # first time only: Airflow's database and login
docker compose up -d                # Airflow at http://localhost:8080
```

Outputs land in `data/` on your machine. Don't run the `pipeline` service and the DAG at the same time: both write to `data/`.

## Without Docker

Use **Python 3.11** or 3.12 (3.13 can't install the pinned libraries).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                     # synthetic files only, nothing downloaded
python -m scripts.pipeline --mode sample      # the 5 smallest participants
python -m scripts.pipeline --mode full        # all 66
```

- **Downloads:** zips already in `data/raw/` are reused. With every listed participant and `Demographics.csv` there, OSF isn't contacted at all, so copy zips you already have into `data/raw/` to run offline.

- **Who is processed:** `data.participants` in the config names exactly who, whatever the mode. It's set to the model team's 10: 5 with DBS and 5 without, so each of the 5 test folds holds one of each. Set it to `null` to go back to `--mode sample` (the smallest `data.sample_size`) or `--mode full` (all 66).
- Participants run 4 at a time (`run.parallel_participants`). One failing is skipped and listed in `reports/failed_participants.json` with an alert; more than 20% failing (`data.max_failed_fraction`) stops the run with exit code 1.
- A dataset that fails its checks also stops the run with exit code 1.
- Every setting is in `config/pipeline.yaml`. Email and Slack alerts are on once `.env` is filled in from `.env.example`; without it they're logged and skipped.

## How it works

```text
OSF ──▶ acquire.py: manifest, download + verify (SHA-256, retries)
          │
          ├─ 1. patients    read.py + pipeline.py: Demographics.csv → each person's wrist (more affected side), DBS, bias slices
          │
          ├─ 2. per participant (4 at a time)                                pipeline.process_participant
          │      diary.py   one row per diary hour: labels, wear, medication (nothing dropped)
          │      ledd.py    doses → events with their levodopa equivalent dose (Jost et al. 2023)
          │      sensor.py  for each hour of that wrist: the 7 hourly features, and 10-s windows at 30 Hz
          │                 then for the whole recording: worn flags and the orientation fix
          │      → data/interim/<pid>/ ; the downloaded zip is deleted
          │
          ├─ 3. build       pipeline.build: join everyone → data/processed/
          ├─ 4. checks      checks.py: the contract (stops the run), warnings, bias report
          └─ 5. version     dvc commit + push (the DAG, or dvc repro)
```

### The windows (`sensor.hour_windows`)

The spec is `windows:` in the config (10 s, 30 Hz, 3 axes, `spec_version: 1`). It must match `windows:` in `Model_Pipeline/config/model.yaml`, and the Android app cuts windows the same way. Bump `spec_version` when anything below changes.

1. Split the hour at recording gaps (> 0.05 s). Nothing is filtered or resampled across a gap.
2. Clip to ±8 g, interpolate onto an exact 100 Hz grid starting on a window boundary, and resample to 30 Hz (`resample_poly`; its low-pass removes everything above 15 Hz).
3. Cut whole 10-s windows: `time_s` is a multiple of 10 on the diary clock, 360 per full hour. COPS stamps each sample at the end of its 10 ms (an hour runs `10:00:00.010` to `11:00:00.000`), so a window may start one sample before the first stamp.
4. Keep each window's per-axis mean (for the orientation check), then remove it.

**Wrist:** the more affected side; when only the other wrist was recorded (COPS-24), that one, with `more_affected_side` false.

**Worn:** false where the diary says the watch wasn't worn, or where over a 30-minute block the watch is cold (< 25 °C) and nearly every window is still.

**Orientation**, checked on the sample's two wrists (recorded at the same time): X runs along the arm, Y across the wrist, Z out of the face.

- A watch strapped on turned round reads X (along the arm) mostly above 0. X and Y are flipped back. This is decided **per wear bout** (a new bout starts after 30 minutes off), because a watch can be put back on another way round: COPS-60's were, after day 0.
- The left wrist is the right's mirror image, so its Y is flipped too. Every wrist then reads like a right one worn the usual way.
- Each run compares the chosen wrist with the other one for up to 12 waking hours (`reports/wrist_agreement.csv`), and warns if they still disagree.

### The clock

`time_s` is seconds since the participant's first diary day began (Day `0 days`, 00:00). The diary row `Time 18 hr` is 17:00–18:00, so `hour_index` = day × 24 + 17 = `time_s` // 3600. Doses sit at mid-hour, since the diary only ticks the hour.

## Outputs

`data/processed/`, versioned by DVC:

| File | One row per | Columns |
| --- | --- | --- |
| `hours.csv` | Diary hour, none dropped | `participant_id`, `day`, `hour_start`, `hour_index`, `visit`, `kinesia_text`, `kinesia_score`, `tremor_score`, `wear_status`, `medication`, `dosage_mg`; the 7 features `W_move_mean` … `W_tremor_ratio` with `W_valid_fraction`, `W_status`; `n_windows`, `worn_fraction`; `drop_reason`; bias slices `sex`, `age_group`, `dbs`, `hoehn_yahr`, `pd_subtype` |
| `events.csv` | Dose | `participant_id`, `time_s`, `kind` (`dose`), `ledd`, `source` (`on_time`), `kinesia_score`, `tremor_score` (empty: check-ins are simulated in training), `drug` |
| `patients.csv` | Participant | `participant_id`, `wrist`, `more_affected_side`, `dbs`, `turned_round_share`, `wear_bouts`, `mirrored` |
| `windows/<pid>/` | Window | `time_s.npy` (n,) int64 · `worn.npy` (n,) bool · `signal.npy` (n, 3, 300) float32, g, mean removed. Load with `np.load(..., mmap_mode="r")` |
| `dataset_meta.json` | Run | The window spec, the wrist rule, and counts |

- **Labels** are made by Model_Pipeline from `hours.csv` (`core/targets.py`): the state is the kinesia score −3 to +3, or asleep when `kinesia_text` is `Sleep`; tremor is the tremor score, left out in sleep hours.
- **`drop_reason`** is empty for hours the hourly baselines can use. Otherwise it's the first that applies: `no_label` (No Data), `not_worn`, `no_sensor_data`, `low_valid_fraction` (< 80%), `sensor_nonwear` (< 50% of windows worn). The window model uses every hour, and the `worn` flag per window.
- **The 7 hourly features** (0.1–3 Hz movement and 4–8 Hz tremor from the 100 Hz magnitude, averaged per second, summarised per hour) are for the baselines and drift monitoring, not the window model.

`data/reports/`: `checks.json` (critical problems, warnings, windows per participant), `wrist_agreement.csv`, `label_issues.json` (diary text/score disagreements, unreadable doses), `failed_participants.json`, `bias_report.json` and `bias_report.md`. Only `bias_report.md` names no participant and is committed to Git.

## Checks (`checks.py`)

- **Critical** (stops the run and the DAG, with an alert): the contract `core/data.py` relies on.
  - A participant-hour appears twice, or a score is out of range.
  - A dose has no time or an LEDD below 0.
  - A patient is missing, or their `dbs` isn't true or false.
  - The windows: wrong shape or type, not stepping by whole windows, NaN values, outside the diary hours, or cut with another spec.
- **Warnings** (alert, the run carries on): fewer than 24 usable hours for a participant, under 50% of windows worn, wrists that still disagree after the orientation fix, and diary label issues.
- **Bias** (`bias_report.md`): rows, participants, and the share of off, on, dyskinetic and asleep, per sex, age group, visit, DBS, Hoehn & Yahr stage and PD subtype. The model pipeline checks the model itself per group.

## Sample run (2026-11-14)

The model team's 10 participants (`data.participants`: 5 with DBS, 5 without):

| | |
| --- | --- |
| Hours | 1,080 diary hours, 789 usable by the baselines |
| Windows | 311,695, of which 300,266 worn. 857 hours are complete (360 windows) |
| Labels | 821 state (345 OFF, 152 ON, 52 dyskinetic, 272 asleep) and 523 tremor |
| Doses | 174 |
| Wrists | 6 left and 4 right. COPS-24 had only its left wrist recorded, so it uses that. Watches turned round: all of COPS-3, 4, 25 and 42, most of COPS-60 (re-strapped after day 0) and part of COPS-18 |
| Checks | 0 critical, 7 warnings: COPS-11 has 15 usable hours; small wrist disagreements for COPS-3, 11 and 25 (−0.08 to −0.15 on one axis); COPS-24 can't be compared; 13 diary scores without a label, for kinesia and for tremor |
| Model contract | Model_Pipeline's `load_dataset` loads it and passes `check()` for every input group |
| Data version | Pushed to the DVC remote; `dvc.lock` names it |

## Code

```text
scripts/
├── acquire.py    list OSF files, choose participants, download with retries and checksums
├── read.py       participant zips, the diary file, hourly accelerometer CSVs (pyarrow, with a pandas fallback), demographics
├── diary.py      the diary as one row per hour: labels (text trusted over score), wear, doses
├── ledd.py       doses → events with LEDD, from config/ledd.yaml (every factor cited)
├── sensor.py     windows, wear, orientation, wrist agreement, the 7 hourly features
├── pipeline.py   patients, one participant, the build, failures, the command line
├── checks.py     the contract, warnings, the bias report
└── utils.py      config, file I/O, logging, email and Slack alerts
dags/cops_pipeline_dag.py   the same steps in Airflow
tests/                      one file per module, on synthetic participants (tests/helpers.py)
```

**Tests:** `pytest -q` runs 138 tests on synthetic participants made by `tests/helpers.py`, and nothing is downloaded or sent. `test_dag.py` needs Airflow, so it skips outside Docker; `docker compose run --rm tests` runs it. Coverage of `scripts/` is about 94% (`pytest --cov=scripts --cov-branch`); CI fails below 85%.

## Data versioning (DVC)

One stage, `build_dataset` in `dvc.yaml`. Its inputs are `scripts/` and both config files; its outputs are everything in `processed/` and `reports/`. `raw/` and `interim/` are working files. The remote is `gs://team17-mlops-neu-training-data/cops-dvc` (the bucket in `infra/storage.tf`). Sign in first with `gcloud auth application-default login`.

```bash
dvc repro                        # rebuild if code, config or data changed; updates dvc.lock
git add dvc.lock data/reports/bias_report.md && git commit -m "Data version: <what changed>"
dvc push                         # upload the data
dvc pull                         # elsewhere: get exactly the data of the checked-out commit
```

After an Airflow run, the `dvc_version` task runs `dvc commit -f build_dataset` and tries `dvc push`. It doesn't commit to Git; do the `git add` and `git commit` yourself.

## Airflow

Open <http://localhost:8080> and log in as `airflow` / `airflow` (local test values in `docker-compose.yaml`). Turn `cops_data_pipeline` on, then trigger it with `{"mode": "sample"}` or `{"mode": "full"}`.

Tasks run in this order: `fetch_manifest` → `patients` → `process_participant` (one mapped task per participant, 4 at a time, 2 retries) → `check_failures` (runs even when some participants failed) → `build` → `run_checks` (no retries) → `dvc_version` → `notify_success`. Any failed task sends an alert.

## Contributions

Newest first.

| Date | Who | What changed | PR |
| --- | --- | --- | --- |
| 2026-11-14 | Dhruv Puri | Rebuilt the pipeline for the window model: every diary hour kept (sleep is a label), 10-s 30 Hz windows with worn flags, the more affected wrist with an orientation fix checked on both wrists, dose events wired in, a patients table, and checks against Model_Pipeline's `Dataset` contract. Removed UPDRS, the row-dropping cleaning, `hours_since_dose`, the schema baseline and the model-level bias check. 18 modules became 8. Then: a named list of participants (the 10 for the model team), orientation decided per wear bout, a fallback when only one wrist was recorded or the side is "both", every wrist mirrored to read like a right one, zips already in `data/raw/` reused without downloading, and 3 more COPS drug names in the LEDD table | (this PR) |
| 2026-11-11 | Sam Selvaraj | Added `scripts/ledd.py`: doses as events with their levodopa equivalent dose, from `config/ledd.yaml` (Jost et al. 2023, Table 1, every value cited), with COMT inhibitor boosts and a clear stop on any unknown drug name. Updated requests, pyarrow and pytest for the Dependabot alerts | (this PR) |
| 2026-10-26 | Shaikh Basim | Made the pipeline run with only Docker: `tests` and `pipeline` services, Airflow's slim image, the Linux `AIRFLOW_UID` setup, and a README that starts with the Docker commands | #9 |
| 2026-10-26 | Sam Selvaraj | Corrected the DAG's task count, and made Git check out text files with LF line endings everywhere, so DVC sees the same files on Windows | #7 |
| 2026-10-25 | Ayush Varma | Ran the pipeline on the 5 sample participants with `dvc repro` and Airflow, recorded the first data version, and documented the results, bias findings, Gantt chart and Airflow steps | #6 |
| 2026-10-24 | Sam Selvaraj | Set up DVC, made processing about 7× faster (pyarrow reading, parallel participants), fixed one failed participant failing the whole Airflow run, added CI and most of the tests | #5 |
| 2026-10-22 | Shaikh Basim | Fixed the pipeline run: both dose-column layouts, failed participants listed with an alert, the DAG path, an optional `.env`, Airflow pinned, and the admin user | #4 |
| 2026-10-21 | Tejaswini Sarikonda | Built the data pipeline: download, diary parsing, wrist features, cleaning, validation, anomaly alerts, bias slicing, the DAG, the DVC stage, Docker and the tests | #3 |
