"""Splits a Dataset by participant: the fixed test set and the training folds.

Owner: P1 (component B).

The test participants are set aside once, before any model work, and never used
for training, tuning or any choice. The rest are split into participant-grouped
folds, so no one's hours are in both training and validation. The same seed
gives the same split on every machine.
"""
