"""The pipeline's steps, in the order run_model_pipeline.py runs them.

Each step is a small class with one job and a run() method. It takes the config,
a tracker and the outputs of earlier steps, writes its own outputs under
artifacts/, logs what it did, and raises a core/errors.py error when it can't go
on, so the run is marked failed and an alert is sent. A step never reads another
step's internals, only its outputs.
"""
