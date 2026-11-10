"""What the model can predict: OFF, tremor and dyskinesia, each defined once.

Owner: P2.

Each target records its kind (binary probability or an ordinal score), its range,
whether it carries an uncertainty interval, and how its hourly label is derived
from the data pipeline's table. Metrics, plots, bias checks and the export read
targets from here, so a definition (such as what counts as OFF) changes in one
place. See MODEL.md, section 2 and Decisions.
"""
