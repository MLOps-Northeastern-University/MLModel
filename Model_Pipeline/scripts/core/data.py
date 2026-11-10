"""What a model is trained and scored on: a Dataset of named input groups.

Owner: P2, with P1 (who fills it in data/loaders.py).

A Dataset holds one row per sample (participant and hour), the input groups
(features, windows, medication, mask), the hourly labels per target, the
participant of each row for grouped folds, and the slice columns for bias checks.
Input groups are loaded only when a model asks for them, and each has a schema
(columns and order, units, window length and rate) that models check, so the
pipeline, phone and Backend stay in step.
"""
