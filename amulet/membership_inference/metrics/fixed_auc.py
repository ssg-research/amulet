import numpy as np
from sklearn import metrics


def get_fixed_auc(preds, true_labels):
    some_stats = {}
    fpr, tpr, _ = metrics.roc_curve(true_labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    low = tpr[np.where(fpr < 0.01)[0][-1]]

    some_stats["fix_auc"] = auc
    some_stats["fix_acc"] = acc
    some_stats["fix_TPR"] = low

    return some_stats
