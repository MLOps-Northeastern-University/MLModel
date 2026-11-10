"""Checks the model across slices of patients and mitigates bias.

Owner: P3 (component H).

Compares each target's metrics across the slices in bias.slice_columns, fails
when a gap is over the threshold, retrains with sample weights when it does, and
writes bias.json and bias_report.md.
"""
