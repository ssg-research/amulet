import numpy as np
import pytest

from amulet.membership_inference.metrics.compute_mi_metrics import compute_mi_metrics


def test_compute_mi_metrics_perfect_separation():
    # members (1) have higher scores than non-members (0)
    preds = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([1, 1, 0, 0])

    results = compute_mi_metrics(preds, labels)

    assert results["auc"] == pytest.approx(1.0)
    assert results["balanced_acc"] == pytest.approx(1.0)
    assert results["tpr_at_fpr"] == pytest.approx(
        1.0
    )  # At 0.01 FPR, we can still get 1.0 TPR if separation is perfect


def test_compute_mi_metrics_random():
    # random-ish scores
    preds = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([1, 1, 0, 0])

    results = compute_mi_metrics(preds, labels)

    assert results["auc"] == pytest.approx(0.5)
    assert results["balanced_acc"] == pytest.approx(0.5)


def test_compute_mi_metrics_output_shape():
    preds = np.random.rand(10)
    labels = np.random.randint(0, 2, 10)

    results = compute_mi_metrics(preds, labels)

    expected_keys = {"auc", "balanced_acc", "tpr_at_fpr", "threshold_score"}
    assert set(results.keys()) == expected_keys
    for val in results.values():
        assert isinstance(val, float)


def test_compute_mi_metrics_tpr_at_fpr_boundary():
    # 1 member, 99 non-members.
    # If we set FPR threshold to 0.01, and we have 100 non-members, we allow 1 FP.
    labels = np.concatenate([np.ones(10), np.zeros(100)])
    # Member scores are high, but let's say only 5 are very high.
    preds = np.concatenate([
        np.linspace(0.6, 0.9, 10),  # members
        np.linspace(0.1, 0.5, 100),  # non-members
    ])

    results = compute_mi_metrics(preds, labels, fpr_threshold=0.01)

    assert 0.0 <= results["tpr_at_fpr"] <= 1.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_compute_mi_metrics_auc_within_bounds(seed, assert_within):
    # random scores, binary labels with both classes guaranteed present
    rng = np.random.default_rng(seed)
    preds = rng.random(50)
    labels = np.concatenate([np.zeros(25), np.ones(25)])
    rng.shuffle(labels)

    results = compute_mi_metrics(preds, labels)

    assert_within(results["auc"], 0.0, 1.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_compute_mi_metrics_balanced_acc_within_bounds(seed, assert_within):
    # random scores, binary labels with both classes guaranteed present
    rng = np.random.default_rng(seed)
    preds = rng.random(50)
    labels = np.concatenate([np.zeros(25), np.ones(25)])
    rng.shuffle(labels)

    results = compute_mi_metrics(preds, labels)

    assert_within(results["balanced_acc"], 0.0, 1.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_compute_mi_metrics_tpr_at_fpr_within_bounds(seed, assert_within):
    # random scores, binary labels with both classes guaranteed present
    rng = np.random.default_rng(seed)
    preds = rng.random(50)
    labels = np.concatenate([np.zeros(25), np.ones(25)])
    rng.shuffle(labels)

    results = compute_mi_metrics(preds, labels)

    assert_within(results["tpr_at_fpr"], 0.0, 1.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_compute_mi_metrics_threshold_score_finite_or_inf(seed):
    # random scores, binary labels with both classes guaranteed present
    rng = np.random.default_rng(seed)
    preds = rng.random(50)
    labels = np.concatenate([np.zeros(25), np.ones(25)])
    rng.shuffle(labels)

    results = compute_mi_metrics(preds, labels)

    # threshold_score is a threshold on arbitrary membership scores, so it
    # is either a finite real or +inf (sklearn roc_curve prepends thresholds[0] =
    # +inf, the synthetic always-predict-negative operating point where fpr=0 and
    # tpr=0). It is never NaN and never -inf.
    score = results["threshold_score"]
    assert np.isfinite(score) or score == np.inf


def test_compute_mi_metrics_threshold_score_forces_inf():
    # all-equal preds mean no real threshold can beat the leading +inf
    # operating point on balanced accuracy, so argmax must select thresholds[0].
    preds = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([1, 1, 0, 0])

    results = compute_mi_metrics(preds, labels)

    assert results["threshold_score"] == np.inf
