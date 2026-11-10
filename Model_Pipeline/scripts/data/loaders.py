"""Reads the data pipeline's outputs into a Dataset (core/data.py).

Owner: P1 (component A).

Reads paths.features_csv, and any other inputs a model asks for, checks them
against their schemas, derives each target's hourly label as core/targets.py
says, and records the data version (the DVC hash) for tracking.
"""
