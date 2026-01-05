import numpy as np
from sklearn import metrics


def compute_mi_metrics(
    preds: np.ndarray,
    true_labels: np.ndarray,
    fpr_threshold: float = 0.01,
) -> dict[str, float]:
    """
    Compute key metrics for membership inference attack evaluation.

    Args:
        preds (np.ndarray): Predicted membership scores or probabilities (higher means more likely member).
        true_labels (np.ndarray): True membership labels (1 for member, 0 for non-member).
        fpr_threshold (float): False Positive Rate threshold at which to compute True Positive Rate (default 0.01 for 1%).

    Returns:
        dict: Dictionary containing:
            - 'auc': Area Under ROC Curve.
            - 'balanced_acc': Balanced accuracy.
            - 'tpr_at_fpr': True Positive Rate at specified false positive rate threshold.
    """
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, preds, pos_label=1)
    auc_score = metrics.auc(fpr, tpr)

    # Balanced accuracy requires binary predictions, use threshold=0.5 by default
    binary_preds = (preds >= 0.5).astype(int)
    balanced_acc = metrics.balanced_accuracy_score(true_labels, binary_preds)

    # Find the TPR at or just below the given FPR threshold
    # Defensive check in case no fpr < threshold (should rarely happen)
    valid_indices = np.where(fpr <= fpr_threshold)[0]
    if len(valid_indices) == 0:
        tpr_at_fpr = 0.0
    else:
        tpr_at_fpr = tpr[valid_indices[-1]]

    return {
        "auc": auc_score,
        "balanced_acc": balanced_acc,
        "tpr_at_fpr": tpr_at_fpr,
    }
