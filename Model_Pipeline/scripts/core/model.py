"""The contract every model follows, and the predictions it returns.

Owner: P2.

A model declares its family, the targets it produces, the input groups it needs,
and whether it scores per hour or per window. It can fit (on training patients
only, with an optional validation set, sample weights and a seed), predict, save
and load itself with a manifest, list its tuning search space, and export.

Predictions always have an hour level, which is what's checked against the diary,
and may have a 10-minute interval level, which is what the dashboard draws.
One model with several targets and several models combined (models/composite.py)
look the same to everything downstream.
"""
