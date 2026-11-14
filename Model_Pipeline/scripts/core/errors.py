"""The pipeline's error types, one per kind of failure.

Owner: P2; add a type here before raising a new kind of failure.

Each step catches what it can recover from and raises one of these for what it
can't, so the run command can mark the run failed, log why, and alert with the
right message: bad configuration, data that doesn't match its schema, a model
used before it's fitted, a failed gate (validation, bias, comparison with
production), or an export that can't be made.
"""


class PipelineError(Exception):
    """Base class: every failure the pipeline raises on purpose."""


class ConfigError(PipelineError):
    """The configuration is missing a value or has one that can't be used."""


class SchemaError(PipelineError):
    """Data doesn't have the columns, types or values it should."""


class NotFittedError(PipelineError):
    """A model was used to predict, save or export before it was fitted."""


class GateError(PipelineError):
    """A gate failed: validation, bias, or comparison with production.

    Attributes:
        gate: Which gate failed, such as "validation" or "bias".
    """

    def __init__(self, gate, message):
        super().__init__(f"{gate} gate failed: {message}")
        self.gate = gate


class ModelError(PipelineError):
    """A model broke its contract (bad predictions, a manifest that doesn't match), or was used wrongly."""


class TrackingError(PipelineError):
    """Tracking was used wrongly: no active run, a bad metric name or value."""


class ExportError(PipelineError):
    """A model couldn't be exported for the phone."""
