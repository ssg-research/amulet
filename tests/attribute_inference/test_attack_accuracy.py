import numpy as np
import pytest

from amulet.attribute_inference.metrics.attack_accuracy import (
    evaluate_attribute_inference,
)


def test_evaluate_attribute_inference_identity():
    # 2 attributes, perfect predictions
    z_test = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])

    predictions = {
        0: {
            "predictions": z_test[:, 0],
            "confidence_values": z_test[:, 0].astype(float),
        },
        1: {
            "predictions": z_test[:, 1],
            "confidence_values": z_test[:, 1].astype(float),
        },
    }

    results = evaluate_attribute_inference(z_test, predictions)

    assert results[0]["attack_accuracy"] == pytest.approx(1.0)
    assert results[0]["auc_score"] == pytest.approx(1.0)
    assert results[1]["attack_accuracy"] == pytest.approx(1.0)
    assert results[1]["auc_score"] == pytest.approx(1.0)


def test_evaluate_attribute_inference_output_shape():
    z_test = np.random.randint(0, 2, (10, 3))
    predictions = {
        i: {
            "predictions": np.random.randint(0, 2, 10),
            "confidence_values": np.random.rand(10),
        }
        for i in range(3)
    }

    results = evaluate_attribute_inference(z_test, predictions)

    assert len(results) == 3
    for i in range(3):
        assert "attack_accuracy" in results[i]
        assert "auc_score" in results[i]
        assert isinstance(results[i]["attack_accuracy"], float)
        assert isinstance(results[i]["auc_score"], float)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_evaluate_attribute_inference_bounds_random(seed, assert_within):
    # random binary z_test with both classes guaranteed per attribute,
    # plus random predictions/confidences, so the metric isn't fed a tautological input.
    rng = np.random.default_rng(seed)
    num_attributes = 3
    n = 50

    z_test = rng.integers(0, 2, (n, num_attributes))
    # Force both classes to be present in every column, regardless of the draw above.
    z_test[0, :] = 0
    z_test[1, :] = 1

    predictions = {
        i: {
            "predictions": rng.integers(0, 2, n),
            "confidence_values": rng.random(n),
        }
        for i in range(num_attributes)
    }

    results = evaluate_attribute_inference(z_test, predictions)

    for i in range(num_attributes):
        assert_within(results[i]["attack_accuracy"], 0.0, 1.0)
        assert_within(results[i]["auc_score"], 0.0, 1.0)
