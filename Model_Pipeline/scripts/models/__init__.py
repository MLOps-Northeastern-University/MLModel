"""Model families, each in its own file and each following core/model.py.

Owner: P2.

To add a family: write <family>.py, register it in registry.py, put its settings
and search space in config/model.yaml, and add tests/models/test_<family>.py.
Import heavy libraries (PyTorch, TensorFlow, LightGBM) inside the family's file
only, so other families don't need them installed.
"""
