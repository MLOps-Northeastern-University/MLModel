# Data Pipeline

Turns the COPS Parkinson's dataset ([osf.io/5xvwn](https://osf.io/5xvwn/)) into an hourly table of wrist-movement features and symptom labels, ready for modelling.

**Target:** hourly kinesia state from the patient's diary (`KinesiaScore`, −3 to +3):

| KinesiaScore | Class |
| --- | --- |
| −3, −2, −1 | `off` |
| 0 | `on` |
| +1, +2, +3 | `dyskinetic` |

## Setup

Use **Python 3.11**. Newer versions fail to install the pinned libraries.

```bash
conda create -n cops python=3.11 -y
conda activate cops
pip install -r requirements.txt
pytest -v
```

All tests should pass. They use small synthetic files, so no download is needed.

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
- Each participant takes a few minutes. Progress is logged every 20 hours.
- All settings are in `config/pipeline.yaml`.
- A participant that fails is skipped and listed in `data/reports/failed_participants.json`, and an alert is sent. If more than 20% fail (`data.max_failed_fraction`), the run stops with exit code 1.

### Airflow

The same steps run as an Airflow DAG in `dags/cops_pipeline_dag.py`. From this folder:

```bash
docker compose up airflow-init   # first time only: sets up the database and the login
docker compose up -d             # starts the webserver and scheduler
```

Open <http://localhost:8080> and log in as `airflow` / `airflow`. Trigger `cops_data_pipeline` with `{"mode": "sample"}` or `{"mode": "full"}`.

A `.env` file is optional. Copy `.env.example` to `.env` only to turn on email or Slack alerts. Without it, alerts are skipped and logged.

## Test run so far

The full dataset is very large (hundreds of hourly sensor files per participant), so the pipeline has so far been run on **one participant, COPS-7**, downloaded manually from OSF and run with `--local-zips`. All current outputs describe COPS-7 only:

- 192 diary hours, 105 usable after cleaning (58 dyskinetic, 31 on, 16 off)
- Removed: 80 without a label (sleep, no data), 5 not worn, 2 with too little valid sensor data
- 0 validation failures; 4 anomaly warnings, all diary text/score mismatches
- The bias report has data-level slices only. The model-level comparison needs several participants and was skipped.

Results across participants will come from a `sample` or `full` run.

## Preprocessing steps

1. **Acquire:** list files on OSF, pick participants, download one zip at a time, and delete it after processing.
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
| `reports/validation_report.json` | Failed checks (empty means all passed) |
| `reports/stats_history.csv` | One summary line per run |
| `reports/anomalies.json` | Critical anomalies and warnings |
| `reports/bias_report.md` | Bias slicing results |
| `raw/manifest.json` | Files used in the run |

Raw and interim data are not committed to Git.
