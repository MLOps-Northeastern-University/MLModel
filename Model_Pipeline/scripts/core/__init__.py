"""The pipeline's contracts: what a target, a dataset, a model and a tracker are.

Everything else depends on these, and these depend on nothing heavy: no MLflow,
scikit-learn, PyTorch or TensorFlow. Change one only by team agreement, in the
README's Interfaces section first.
"""
