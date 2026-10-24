# Data Pipeline

Turns the COPS Parkinson's dataset ([osf.io/5xvwn](https://osf.io/5xvwn/)) into an hourly table of wrist-movement features and symptom labels, ready for modelling.

**Target:** hourly kinesia state from the patient's diary (`KinesiaScore`, −3 to +3):

| KinesiaScore | Class |
| --- | --- |
| −3, −2, −1 | `off` |
| 0 | `on` |
| +1, +2, +3 | `dyskinetic` |

## Dataset

**COPS** ([osf.io/5xvwn](https://osf.io/5xvwn/)), a public dataset of people with Parkinson's disease recorded at home. Check the OSF page for its licence, and cite it as the page asks.

- **66 participants**, one zip each (47.9 GB in all; the 5 smallest, used by `sample` mode, are 1.1 GB).
- **Wrist sensors** on both wrists: one CSV per wrist per hour, 100 samples a second, with acceleration `X`, `Y`, `Z` (in g), `Photo` (light) and `Temp` (temperature).
- **An hourly symptom diary** filled in by the participant: kinesia (−3 to +3, the label), tremor, freezing, falls, medication and dose, and whether the watch was worn.
- **Clinical exams:** MDS-UPDRS Part III, OFF and ON medication.
- **`Demographics.csv`:** age, sex, handedness, Parkinson's subtype and dominant side, Hoehn & Yahr stage, years since diagnosis, and DBS.

Participant IDs are the dataset's own pseudonyms (`COPS-7`). Even so, files that list participants are versioned by DVC, not committed (see [Data versioning](#data-versioning-dvc)).

## Setup

Use **Python 3.11**. Newer versions fail to install the pinned libraries.

```bash
conda create -n cops python=3.11 -y
conda activate cops
pip install -r requirements.txt
pytest -v
```

All tests should pass. They use small synthetic files, so no download is needed.

## Tests

`pytest` runs 116 tests in `tests/`, on synthetic participants made by `tests/helpers.py`. No test downloads anything or sends a real alert.

| File | What it checks |
| --- | --- |
| `test_diary.py` | Diary parsing, kinesia classes, text/score mismatches, every dose-column layout |
| `test_features.py` | Movement and tremor features, gravity removal, missing and impossible values, corrupt files, and the fast CSV reader matching pandas exactly |
| `test_clean_and_join.py` | Every cleaning rule, duplicates, merging participants and demographics |
| `test_validate_and_anomalies.py` | Rules, schema against the baseline, statistics, outliers, label issues |
| `test_acquire_and_bias.py` | Choosing participants from the OSF listing, and the bias report |
| `test_download.py` | Retries after a dropped connection, refusing corrupt or truncated downloads, the download cache, OSF paging and outages, local zips, downloading only `https://` from OSF, and never choosing an odd file name |
| `test_robustness.py` | Diaries in UTF-8, with a BOM or in Windows encoding; wrong separators, empty and cut-off files; partial failures (one participant failing doesn't stop the run, too many do); config files; passwords never logged |
| `test_alerts.py` | Email and Slack alerts when set up, skipped when not, and a broken mail server or webhook never crashing the run |
| `test_run_pipeline.py` | Failed participants reported, the failure limit, parallel runs matching sequential ones |
| `test_end_to_end.py` | The whole pipeline on four synthetic participants: every output, exact row counts, a second run checked against the first, and stopping on too many failures or on critical anomalies |
| `test_dag.py` | The Airflow DAG loads, its tasks run in order, failures retry and alert, and one failed participant doesn't fail the run. Needs Airflow, so it skips locally; run it in the Airflow image: `docker compose run --rm -w /opt/airflow/repo/Data_Pipeline airflow-scheduler python -m pytest tests/test_dag.py` |

Coverage of `scripts/` is about 92%: `pytest --cov=scripts --cov-branch`. CI fails below 85%.

## Running

Run from this folder.

```bash
# Download the 5 smallest participants from OSF and run everything
python -m scripts.run_pipeline --mode sample

# All 66 participants
python -m scripts.run_pipeline --mode full

# Use participant zips you already downloaded
python -m scripts.run_pipeline --local-zips C:\path\to\folder
```

- Write `scripts.run_pipeline` without `.py`.
- For `--local-zips`, the folder needs the original zips, named like `COPS-7.zip`, plus `Demographics.csv` from the top level of the OSF project.
- Processing one participant takes about half a minute, plus its download (see [Speed](#speed)). Progress is logged every 20 hours.
- Participants run 4 at a time (`run.parallel_participants`), each in its own process. Set it to 1 to run them one after another.
- All settings are in `config/pipeline.yaml`.
- A participant that fails is skipped and listed in `data/reports/failed_participants.json`, and an alert is sent. If more than 20% fail (`data.max_failed_fraction`), the run stops with exit code 1. Airflow follows the same rule, in its `check_failures` task.

### Airflow

The same steps run as an Airflow DAG in `dags/cops_pipeline_dag.py`. From this folder:

```bash
docker compose up airflow-init   # first time only: sets up the database and the login
docker compose up -d             # starts the webserver and scheduler
```

Open <http://localhost:8080> and log in as `airflow` / `airflow`. Trigger `cops_data_pipeline` with `{"mode": "sample"}` or `{"mode": "full"}`.

A `.env` file is optional. Copy `.env.example` to `.env` only to turn on email or Slack alerts. Without it, alerts are skipped and logged.

## Test run so far

The full dataset is very large (hundreds of hourly sensor files per participant), so the pipeline has so far been run on **single participants**, downloaded manually from OSF and run with `--local-zips`. The first run, on **COPS-7**:

- 192 diary hours, 105 usable after cleaning (58 dyskinetic, 31 on, 16 off)
- Removed: 80 without a label (sleep, no data), 5 not worn, 2 with too little valid sensor data
- 0 validation failures; 4 anomaly warnings, all diary text/score mismatches
- The bias report has data-level slices only. The model-level comparison needs several participants and was skipped.

The summary reports in Git (`data/reports/cleaning_summary.json`, `bias_report.md`) are from a later single-participant run, on COPS-25: 72 diary hours, 36 usable. Results across participants will come from a `sample` or `full` run.

## Preprocessing steps

1. **Acquire:** list files on OSF, pick participants, and for each one download its zip (checked against OSF's SHA-256, with retries), process it, and delete it. Several participants run at once.
2. **Parse the diary:** read the hourly diary, map `KinesiaScore` to off / on / dyskinetic, and flag rows where the text and score disagree (the text is trusted).
3. **Extract features** for each hour and each wrist:
    1. Magnitude per row: √(X² + Y² + Z²)
    2. Band-pass filter into 0.1–3 Hz (movement, also removes gravity) and 4–8 Hz (tremor)
    3. Absolute value, averaged per second
    4. Hourly summary: `move_mean`, `move_std`, `move_p95`, `still_fraction`, `tremor_mean`, `tremor_p95`, `tremor_ratio`
4. **Combine wrists:** prefix features with `L_` and `R_`, and add left-right asymmetry (`asym_move_mean`, `asym_tremor_mean`, `asym_still_fraction`).
5. **Add patient info:** UPDRS-III off and on scores, levodopa response, demographics, and hours since the last dose.
6. **Clean:** remove hours with no label (sleep, no data), not worn, missing or low-quality sensor data, and duplicates.
7. **Validate:** check fixed rules, compare with the baseline schema, and save statistics.
8. **Detect anomalies:** missing values, outliers, low coverage, and label issues. Alerts are sent by email or Slack if set up in `.env`.
9. **Bias report:** compare label balance and a baseline model's results across sex, age group, visit, DBS, Hoehn & Yahr, and PD subtype.

## Output files

All in `data/`.

| File | Contents |
| --- | --- |
| `processed/hourly_features.csv` | **Final dataset for modelling:** one row per usable participant-hour |
| `processed/hourly_all.csv` | All diary hours before cleaning |
| `processed/label_issues.json` | Diary text/score mismatches and missing UPDRS files |
| `interim/COPS-X_hourly.csv` | Per-participant table before merging |
| `reports/failed_participants.json` | Participants that could not be processed, with the error (empty means none) |
| `reports/cleaning_summary.json` | Hours kept and removed, by reason |
| `reports/schema_baseline.json` | Reference schema from the first run |
| `reports/schema_latest.json` | Schema of the latest run |
| `reports/validation_report.json` | Failed checks (empty means all passed) |
| `reports/stats/stats_<run>.json` | Full statistics of each run |
| `reports/stats_history.csv` | One summary line per run |
| `reports/anomalies.json` | Critical anomalies and warnings |
| `reports/bias_report.md` | Bias slicing results, to read |
| `reports/bias_report.json` | The same, as data, with the participants the model was tested on |
| `raw/manifest.json` | Files used in the run |

Only `cleaning_summary.json` and `bias_report.md` are committed to Git; DVC versions the rest of `processed/` and `reports/`. Raw and interim data aren't versioned.

## Data card

`processed/hourly_features.csv`: one row per participant-hour that has a label and usable sensor data from both wrists.

| Group | Columns | What they are |
| --- | --- | --- |
| Identifiers | `participant_id`, `visit`, `day`, `hour_start`, `hour_index` | Who and when: the diary day, the hour it starts (0–23), and hours since the first diary day (`day × 24 + hour_start`) |
| **Label** | `kinesia_class` | `off`, `on` or `dyskinetic`, from `kinesia_score` |
| Diary | `kinesia_text`, `kinesia_score`, `tremor_score`, `tremor_present`, `freezing`, `fall`, `wear_status` | The participant's own hourly entries; `kinesia_score` runs from −3 to +3 |
| Medication | `medication_taken`, `medication`, `dosage_mg`, `hours_since_dose` | Doses logged in the diary, and hours since the last one |
| Movement, per wrist (`L_`, `R_`) | `move_mean`, `move_std`, `move_p95`, `still_fraction` | Everyday movement (0.1–3 Hz), from the per-second averages over the hour |
| Tremor, per wrist | `tremor_mean`, `tremor_p95`, `tremor_ratio` | Tremor-band movement (4–8 Hz), and its size relative to everyday movement |
| Quality, per wrist | `n_samples`, `valid_fraction`, `n_gaps`, `n_clipped`, `temp_mean`, `photo_mean`, `status` | How much usable data the hour had, gaps and clipped values, and the sensor's temperature and light |
| Asymmetry | `asym_move_mean`, `asym_tremor_mean`, `asym_still_fraction` | Right against left: +1 all on the right, −1 all on the left |
| Clinical | `updrs_off_total`, `updrs_on_total`, `levodopa_response`, `updrs_days_since_exam`, `has_updrs_off`, `has_updrs_on` | MDS-UPDRS Part III OFF and ON; `levodopa_response` is (OFF − ON) / OFF |
| Demographics | `age`, `age_group`, `sex`, `handedness`, `pd_subtype`, `pd_dominantside`, `hoehn_yahr`, `pd_yearssincediagnosis`, `dbs` | From `Demographics.csv`; `age_group` is `<55`, `55-64` or `65+`. Also the slices for the bias report. |
| Files | `left_zip`, `left_csv`, `right_zip`, `right_csv` | The sensor files each row was computed from, for tracing |

## Code structure

```text
Data_Pipeline/
├── config/pipeline.yaml   every setting: paths, download, features, cleaning, validation, alerts, bias
├── dags/                  cops_pipeline_dag.py: the Airflow DAG
├── scripts/               the pipeline, one module per step
│   ├── acquire.py         list OSF files, choose participants, download with retries and checksums
│   ├── diary.py           read the hourly diary, map kinesia to classes, flag text/score mismatches
│   ├── features.py        read the wrist CSVs, filter, and compute the hourly features per wrist
│   ├── patient_info.py    UPDRS exams and levodopa response
│   ├── process_patient.py one participant zip -> one hourly table (the unit of parallel work)
│   ├── join.py            merge participants and demographics
│   ├── clean.py           drop unusable hours, with a count for each reason
│   ├── validate.py        rules, schema against the baseline, statistics and their history
│   ├── anomalies.py       missing values, outliers, low coverage, label issues
│   ├── bias.py            data slicing and the baseline model's results per slice, with reweighting
│   ├── alerts.py          email and Slack alerts
│   ├── pipeline_steps.py  each step as a function, shared by the DAG and the command line
│   ├── run_pipeline.py    the whole pipeline without Airflow (also what dvc repro runs)
│   └── utils/             config loading, file reading and writing, logging
├── tests/                 pytest tests on small synthetic files (no download)
├── data/                  raw/, interim/, processed/, reports/ (see Output files)
├── docs/gantt/            Airflow Gantt charts
├── logs/                  pipeline.log
├── dvc.yaml               the DVC stage
├── Dockerfile, docker-compose.yaml   the Airflow setup
└── requirements.txt
```

## Data versioning (DVC)

The pipeline is one DVC stage, `build_dataset` in `dvc.yaml`: its inputs are `scripts/` and `config/pipeline.yaml`, and its outputs are everything it writes to `processed/` and `reports/` (listed in `dvc.yaml`). `raw/` and `interim/` are working files and aren't versioned. DVC is set up in this folder (`dvc init --subdir`, so `.dvc/` is here and not at the repo root).

**What goes where:**

| Where | What |
| --- | --- |
| Git | The code, `dvc.yaml`, and `dvc.lock`, which records the hash of every input and output for each version |
| Git, readable by anyone | The summary reports with no participant IDs: `cleaning_summary.json` and `bias_report.md` (`cache: false` in `dvc.yaml`) |
| DVC, never Git | The datasets and the reports that name participants: `processed/`, the statistics, schema, validation, anomaly and failure reports, and `bias_report.json` (it lists the test participants) |

There are no separate `.dvc` files: in a DVC pipeline, `dvc.lock` records the outputs, doing the job `.dvc` files do for data added with `dvc add`.

The schema baseline, the statistics and their history are `persist: true`, so they're kept between runs: each run is validated against the first, and the history grows by one line per run.

**Making a version:**

```bash
dvc repro            # runs the pipeline if code, config or data changed, and updates dvc.lock
git add dvc.lock data/reports/cleaning_summary.json data/reports/bias_report.md
git commit -m "Data version: <what changed>"
dvc push             # uploads the data to the remote
```

After an Airflow run, the `dvc_version` task does the same with `dvc commit -f build_dataset`.

**Getting a version:** `git checkout <commit>` and then `dvc pull` restore exactly the data of that commit. Without access to the remote, `dvc repro` rebuilds it from OSF, and `dvc.lock` shows whether the result matches.

**Where the data is stored (the remote).** The data goes to a DVC remote, set once per machine. It isn't set in the repo yet, so choose one:

```bash
# Google Cloud Storage (dvc[gs] is in requirements.txt). Sign in first: gcloud auth application-default login
dvc remote add -d storage gs://<bucket>/cops-dvc

# Or a folder outside the repo, until a bucket exists
dvc remote add -d --local storage <path outside the repo>
```

`--local` keeps a folder remote in `.dvc/config.local`, which is not committed, so each person's path stays their own. A bucket remote is committed in `.dvc/config` for everyone. Moving from a folder to a bucket later needs no re-run: add the bucket remote, then `dvc push`.

## Speed

Each participant has about 190 diary hours, and each hour is one CSV per wrist of 360,000 rows (100 a second). Profiling one full-size hour showed where the time went: reading the CSV (0.25 s) and converting its clock times to seconds (0.20 s), not the filtering (0.015 s). Two changes followed:

| Change | Before | After |
| --- | --- | --- |
| Read each CSV with pyarrow, with typed columns, so clock times are parsed while reading. Files with a broken value fall back to pandas and are cleaned as before; the features are identical. | 0.68 s per wrist-hour | 0.09 s per wrist-hour |
| Process participants in parallel on the command line too, not only in Airflow (`run.parallel_participants`, 4) | One participant at a time | 4 at a time |

Together, processing one participant went from about 4.3 minutes to about 35 seconds, and the command-line run handles four at once. Downloading is now the larger share of the time. Measured on one laptop with a synthetic full-size hour; real runs also depend on the network.

## Contributions

Newest first.

| Date | Who | What changed | PR |
| --- | --- | --- | --- |
| 2026-10-22 | Shaikh Basim | Fixed the pipeline run: diaries with `Dosage` or `Dosage_1..N` columns are read, failed participants are listed in `failed_participants.json` with an alert, the Airflow DAG path in `docker-compose.yaml` is fixed, `.env` is optional, Airflow is pinned in the Dockerfile, and `airflow-init` creates the admin user | #4 |
| 2026-10-21 | Tejaswini Sarikonda | Built the data pipeline: download from OSF, diary parsing, wrist features, cleaning, validation and statistics, anomaly alerts, bias slicing, the Airflow DAG, the DVC stage, the Docker setup and the tests | #3 |
