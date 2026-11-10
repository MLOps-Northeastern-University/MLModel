"""Exports the selected model and builds the release bundle.

Owner: P4 (component J).

Writes the model file, its metadata, the training inputs' statistics for drift,
the metrics and bias results, and test cases (inputs with the expected outputs)
that the phone and Backend check before using the model.
"""
