"""Tunes each model's settings with Optuna.

Owner: P2 (component D).

Searches the space each family declares (config/model.yaml), scoring every trial
by the same participant-grouped folds on training participants only. Logs each
trial as a child run and passes the best settings to train.
"""
