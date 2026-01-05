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
            - 'balanced_acc': Maximum balanced accuracy over all thresholds.
            - 'tpr_at_fpr': True Positive Rate at specified false positive rate threshold.
    """
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, preds, pos_label=1)
    auc_score = metrics.auc(fpr, tpr)

    # Maximum balanced accuracy across all thresholds
    # Balanced accuracy = (TPR + TNR) / 2 = 1 - (FPR + (1-TPR)) / 2
    balanced_accs = 1 - (fpr + (1 - tpr)) / 2
    best_idx = np.argmax(balanced_accs)
    threshold_score = thresholds[best_idx]
    max_balanced_acc = np.max(balanced_accs)

    # Find the TPR at or just below the given FPR threshold
    valid_indices = np.where(fpr <= fpr_threshold)[0]
    if len(valid_indices) == 0:
        tpr_at_fpr = 0.0
    else:
        tpr_at_fpr = tpr[valid_indices[-1]]

    return {
        "auc": float(auc_score),
        "balanced_acc": float(max_balanced_acc),
        "tpr_at_fpr": float(tpr_at_fpr),
        "threshold_score": float(threshold_score),
    }
