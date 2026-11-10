"""Scores the predictions and applies the validation gate.

Owner: P1 (component F).

Computes each target's metrics on the out-of-fold and test predictions, per
participant and overall, with non-DBS patients reported separately, writes
metrics.json and the plots, and fails the gate when a metric is below its
threshold in config/model.yaml.
"""
