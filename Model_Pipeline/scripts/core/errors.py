"""The pipeline's error types, one per kind of failure.

Owner: P2; add a type here before raising a new kind of failure.

Each step catches what it can recover from and raises one of these for what it
can't, so the run command can mark the run failed, log why, and alert with the
right message: bad configuration, data that doesn't match its schema, a model
used before it's fitted, a failed gate (validation, bias, comparison with
production), or an export that can't be made.
"""
