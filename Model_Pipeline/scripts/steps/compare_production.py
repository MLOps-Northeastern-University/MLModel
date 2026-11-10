"""Compares the new model with the one in production.

Owner: P4 (component K).

On the same fixed test set: the new model goes ahead only if it scores better and
has no larger fairness gap. Otherwise the run stops and the current model stays.
"""
