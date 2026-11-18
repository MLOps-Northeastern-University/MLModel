# The training VM

CI makes one VM for each training run and deletes it after. [`startup.sh`](startup.sh) is the VM's start-up script: it sets the machine up, runs one DAG, saves the result and shuts down. The VMs run in their own project (Sam's, for its GPU quota and credits), and use the team project's buckets, MLflow, images and secrets. What every VM shares is in Terraform: its account, its network and CI's permission to make it in [`../../infra/compute/`](../../infra/compute/); what it may use in the team project in [`../../infra/training_vm_access.tf`](../../infra/training_vm_access.tf) and [`secrets.tf`](../../infra/secrets.tf).

## What a run does

1. **The machine:** Docker; for a GPU run, the NVIDIA driver (its installer reboots once) and NVIDIA's container toolkit.
2. **The code:** the repo at the run's commit, and the Airflow, data and model images CI pushed to Artifact Registry.
3. **The files** (a model run): the data the commit's `dvc.lock` names (`dvc pull`), harnet10's weights, and what earlier runs kept.
4. **Airflow:** started with one login per teammate (the `airflow-users` secret). Its REST API is off and logins are rate-limited.
5. **The run:** the DAG triggered with the run's settings. A Cloudflare tunnel serves the Airflow UI at a random `https://….trycloudflare.com` link, emailed to the team (never printed in a public log). The VM opens no port.
6. **After:** what runs keep, the Airflow logs and `result.json` go back to GCS, and the VM shuts down. If anything fails on the way, the run is recorded as failed and the VM still shuts down.

## The run's settings

CI passes them as VM metadata:

| Key | What |
|---|---|
| `startup-script` | this file |
| `mednova-run-id` | the run's name, unique (`gh-<GitHub run id>-<attempt>`) |
| `mednova-git-sha` | the commit |
| `mednova-dag` | `model_pipeline` or `cops_data_pipeline` |
| `mednova-dag-conf` | the DAG's trigger config, JSON (from `retrain.yaml`, or the drift alert) |
| `mednova-airflow-image`, `mednova-data-image`, `mednova-model-image` | the images |
| `mednova-gpu` | `1` for a GPU run |
| `mednova-mlflow-uri` | the team MLflow |
| `mednova-team-project` | the team project, whose buckets, secrets and images it uses |

## What it keeps in GCS

All in `gs://team17-mlops-neu-training-data/`, private:

| Folder | What | Written by |
|---|---|---|
| `cops-dvc/` | the data, by DVC | the data DAG |
| `harnet10/` | harnet10's weights (our pinned copy; its licence allows no redistribution) | by hand, once |
| `state/` | what one run leaves the next: harnet10's feature cache and the pretrained bodies (`cache/`), the tuning studies, `tuned_params.json`, the test-scored markers | each run |
| `runs/<run id>/` | `result.json` (CI waits for it), the start-up log, the Airflow logs | each run |

## Getting in during a run

- **The Airflow UI:** the link in the team's email, with your Airflow login. Your password: `gcloud secrets versions access latest --secret airflow-users --project team17-mlops-neu`.
- **A shell, to debug:** `gcloud compute ssh <vm> --zone <zone> --tunnel-through-iap`. The network lets in nothing else.
- **The start-up script's log:** `gcloud compute instances get-serial-port-output <vm> --zone <zone>`, or `runs/<run id>/startup.log` once it's done.
