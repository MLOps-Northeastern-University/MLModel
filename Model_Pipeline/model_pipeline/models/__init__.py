"""Model families, each following core/model.py.

Owner: P2.

    window_classifier.py        window_logistic and window_boosting: each window on its own, no past
    motor_causal_transformer/   the window model: attention over hours of past

To add a family: write <family>.py (or a package, for a large one), register it in
registry.py, put its settings and search space in config/model.yaml, and add its
tests under tests/models/.
Import heavy libraries (PyTorch, TensorFlow, LightGBM) inside the family's file
only, so other families don't need them installed.
"""
