"""Attack accuracy for attribute inference attacks"""

from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import numpy as np


def evaluate_attribute_inference(
    z_test: np.ndarray, predictions: dict[int, dict[str, np.ndarray]]
) -> dict[int, dict[str, float]]:
    """
    Calculates the attack accuracy and AUC score
        for an attribute inference attack.

    Attributes:
        z_test: :class: `~np.ndarray`
            The ground truth indicators of the sensitive attributes
        predictions:
            Nested dictionary of predictions where the first key
            is the ith attribute and the second key is a string
            denoting "prediction" or "confidence_values"

    Returns:
        Nested dictionary where the first key is the index of
        the attribute being tested, and the second key is the
        metric being calculated.
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
