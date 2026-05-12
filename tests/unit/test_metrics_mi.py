import numpy as np
import pytest

from amulet.membership_inference.metrics.compute_mi_metrics import compute_mi_metrics


def test_compute_mi_metrics_perfect_separation():
    # Arrange: members (1) have higher scores than non-members (0)
    preds = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([1, 1, 0, 0])

    # Act
    results = compute_mi_metrics(preds, labels)

    # Assert
    assert results["auc"] == pytest.approx(1.0)
    assert results["balanced_acc"] == pytest.approx(1.0)
    assert results["tpr_at_fpr"] == pytest.approx(
        1.0
    )  # At 0.01 FPR, we can still get 1.0 TPR if separation is perfect


def test_compute_mi_metrics_random():
    # Arrange: random-ish scores
    preds = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([1, 1, 0, 0])

    # Act
    results = compute_mi_metrics(preds, labels)

    # Assert
    assert results["auc"] == pytest.approx(0.5)
    assert results["balanced_acc"] == pytest.approx(0.5)


def test_compute_mi_metrics_output_shape():
    # Arrange
    preds = np.random.rand(10)
    labels = np.random.randint(0, 2, 10)

    # Act
    results = compute_mi_metrics(preds, labels)

    # Assert
    expected_keys = {"auc", "balanced_acc", "tpr_at_fpr", "threshold_score"}
    assert set(results.keys()) == expected_keys
    for val in results.values():
        assert isinstance(val, float)


def test_compute_mi_metrics_tpr_at_fpr_boundary():
    # Arrange: 1 member, 99 non-members.
    # If we set FPR threshold to 0.01, and we have 100 non-members, we allow 1 FP.
    labels = np.concatenate([np.ones(10), np.zeros(100)])
    # Member scores are high, but let's say only 5 are very high.
    preds = np.concatenate([
        np.linspace(0.6, 0.9, 10),  # members
        np.linspace(0.1, 0.5, 100),  # non-members
    ])

    # Act
    results = compute_mi_metrics(preds, labels, fpr_threshold=0.01)

    # Assert
    assert 0.0 <= results["tpr_at_fpr"] <= 1.0
