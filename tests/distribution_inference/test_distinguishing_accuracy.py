import numpy as np
import pytest

from amulet.distribution_inference.metrics.distinguishing_accuracy import (
    evaluate_distinguishing_accuracy,
)


def test_evaluate_distinguishing_accuracy_identity():
    # perfect scores
    predictions = np.array([0.9, 0.1, 0.8, 0.2])
    ground_truth = np.array([1, 0, 1, 0])

    results = evaluate_distinguishing_accuracy(predictions, ground_truth)

    assert results["distinguishing_accuracy"] == pytest.approx(1.0)
    assert results["auc_score"] == pytest.approx(1.0)


def test_evaluate_distinguishing_accuracy_threshold_flip():
    # scores near 0.5
    predictions = np.array([0.51, 0.49])
    ground_truth = np.array([1, 0])

    res_05 = evaluate_distinguishing_accuracy(predictions, ground_truth, threshold=0.5)
    res_06 = evaluate_distinguishing_accuracy(predictions, ground_truth, threshold=0.6)

    assert res_05["distinguishing_accuracy"] == 1.0
    # At 0.6, 0.51 becomes 0. ground_truth is 1. Misclassified.
    # At 0.6, 0.49 becomes 0. ground_truth is 0. Correct.
    # So 0.5 accuracy.
    assert res_06["distinguishing_accuracy"] == 0.5


def test_evaluate_distinguishing_accuracy_output_shape():
    predictions = np.random.rand(10)
    ground_truth = np.random.randint(0, 2, 10)

    results = evaluate_distinguishing_accuracy(predictions, ground_truth)

    assert "distinguishing_accuracy" in results
    assert "auc_score" in results
    assert isinstance(results["distinguishing_accuracy"], float)
    assert isinstance(results["auc_score"], float)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_evaluate_distinguishing_accuracy_outputs_within_unit_interval(
    seed, assert_within
):
    # random scores and binary labels guaranteed to contain both classes
    rng = np.random.default_rng(seed)
    predictions = rng.random(50)
    ground_truth = np.zeros(50, dtype=int)
    ground_truth[:25] = 1
    rng.shuffle(ground_truth)

    results = evaluate_distinguishing_accuracy(predictions, ground_truth)

    assert_within(results["distinguishing_accuracy"], 0.0, 1.0)
    assert_within(results["auc_score"], 0.0, 1.0)
