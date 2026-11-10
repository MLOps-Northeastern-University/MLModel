"""Pushes the bundle to the model registry, and rolls back.

Owner: P4 (component K).

Uploads the bundle to the models bucket, registers it with the Backend, and
records the MLflow run it came from, so any prediction traces back to its code
and data. Rolls back to the previous version on request.
"""
