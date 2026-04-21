import numpy as np
import pytest

from amulet.attribute_inference.metrics.attack_accuracy import (
    evaluate_attribute_inference,
)


def test_evaluate_attribute_inference_identity():
    # Arrange: 2 attributes, perfect predictions
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

    # Act
    results = evaluate_attribute_inference(z_test, predictions)

    # Assert
    assert results[0]["attack_accuracy"] == pytest.approx(1.0)
    assert results[0]["auc_score"] == pytest.approx(1.0)
    assert results[1]["attack_accuracy"] == pytest.approx(1.0)
    assert results[1]["auc_score"] == pytest.approx(1.0)


def test_evaluate_attribute_inference_output_shape():
    # Arrange
    z_test = np.random.randint(0, 2, (10, 3))
    predictions = {
        i: {
            "predictions": np.random.randint(0, 2, 10),
            "confidence_values": np.random.rand(10),
        }
        for i in range(3)
    }

    # Act
    results = evaluate_attribute_inference(z_test, predictions)

    # Assert
    assert len(results) == 3
    for i in range(3):
        assert "attack_accuracy" in results[i]
        assert "auc_score" in results[i]
        assert isinstance(results[i]["attack_accuracy"], float)
        assert isinstance(results[i]["auc_score"], float)
