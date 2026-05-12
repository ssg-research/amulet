"""Attack accuracy for attribute inference attacks"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def evaluate_attribute_inference(
    z_test: np.ndarray, predictions: dict[int, dict[str, np.ndarray]]
) -> dict[int, dict[str, float]]:
    """
    Calculate attack accuracy and AUC for each sensitive attribute.

    Args:
        z_test: Ground truth sensitive attribute labels, shape (N, num_attributes).
        predictions: Nested dict mapping attribute index to a dict with
            "predictions" and "confidence_values" arrays.

    Returns:
        Nested dict mapping attribute index to a dict with keys
        "attack_accuracy" and "auc_score".
    """
    num_attributes = z_test.shape[1]

    metrics = {}

    for i in range(num_attributes):
        metrics[i] = {}
        metrics[i]["attack_accuracy"] = balanced_accuracy_score(
            z_test[:, i], predictions[i]["predictions"]
        )
        metrics[i]["auc_score"] = roc_auc_score(
            z_test[:, i], predictions[i]["confidence_values"]
        )

    return metrics
