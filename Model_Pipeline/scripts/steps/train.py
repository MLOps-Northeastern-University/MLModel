"""Trains the configured models on the training folds.

Owner: P2 (component C).

Fits each model on each fold's training participants, collects the out-of-fold
predictions, refits on all training participants, scores the fixed test set
once, and writes the predictions for evaluate and bias. Runs the shuffled-label
leakage check on the same folds.
"""
