"""Tracking with MLflow.

Owner: P2.

Uses the MLflow server at mlflow.tracking_uri (or MLFLOW_TRACKING_URI), and falls
back to a local mlruns/ folder, with a warning, when the server can't be reached.
Tags every run with the git commit, the data version, the config's hash, the seed
and the library versions, and saves the run's log and config as artifacts.
"""
