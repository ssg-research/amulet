"""Distinguishing accuracy for distribution inference attacks."""

import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_distinguishing_accuracy(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Score a distribution inference attack.

    Args:
        predictions: 1-D array of attack scores in ``[0, 1]``.
        ground_truth: 1-D binary array. 0 for distribution 1, 1 for distribution 2.
        threshold: Decision threshold applied to ``predictions``.

    Returns:
        Dictionary with:
            - ``"distinguishing_accuracy"``: fraction of correct binary decisions.
            - ``"auc_score"``: ROC-AUC of the raw scores.
    """
    binary = (predictions >= threshold).astype(int)
    return {
        "distinguishing_accuracy": float(np.mean(binary == ground_truth)),
        "auc_score": float(roc_auc_score(ground_truth, predictions)),
    }
