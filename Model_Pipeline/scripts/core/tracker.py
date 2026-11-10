"""The contract for experiment tracking: runs, parameters, metrics and artifacts.

Owner: P2.

Steps and models log through a Tracker passed to them, never through MLflow
directly, so tracking can be swapped (tracking/) and tested without a server.
Runs nest: one pipeline run, then a child run per step, model part, fold or
tuning trial. Metric names follow one scheme, {split}/{target}/{metric}, and a
run always ends as finished or failed.
"""
