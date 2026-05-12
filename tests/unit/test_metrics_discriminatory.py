import numpy as np
import pytest

from amulet.discriminatory_behavior.metrics.discriminatory_behavior import (
    DiscriminatoryBehavior,
)


def test_p_rule_divide_by_zero_guard():
    # Arrange: y_z_0 (mean prediction for attribute=0) is 0
    predictions = np.array([1, 0, 0])
    attributes = np.array([1, 0, 0])

    # Act
    result = DiscriminatoryBehavior.p_rule(predictions, attributes)

    # Assert
    assert result == 0.0


def test_p_rule_perfect_parity():
    # Arrange: same positive rate for both groups
    predictions = np.array([1, 1, 0, 0])
    attributes = np.array([1, 0, 1, 0])

    # Act
    result = DiscriminatoryBehavior.p_rule(predictions, attributes)

    # Assert
    assert result == 100.0


@pytest.mark.parametrize(
    "predictions, attributes, expected",
    [
        (np.array([1, 1, 0, 0]), np.array([1, 1, 0, 0]), 1.0),  # Perfect correlation
        (
            np.array([1, 0, 1, 0]),
            np.array([1, 1, 0, 0]),
            0.0,
        ),  # No correlation (50% pos in both)
    ],
)
def test_demographic_parity(predictions, attributes, expected):
    # Act
    result = DiscriminatoryBehavior.demographic_parity(predictions, attributes)

    # Assert
    assert result == pytest.approx(expected, abs=1e-7)


def test_true_positive_parity_identity():
    # Arrange: predictions match targets perfectly
    targets = np.array([1, 1, 1, 1])
    predictions = np.array([1, 1, 1, 1])
    attributes = np.array([1, 1, 0, 0])

    # Act
    result = DiscriminatoryBehavior.true_positive_parity(
        predictions, targets, attributes
    )

    # Assert: P(y_hat=1|y=1,a=1) = 1.0, P(y_hat=1|y=1,a=0) = 1.0. Diff = 0
    assert result == pytest.approx(0.0)


def test_false_positive_parity_identity():
    # Arrange: predictions match targets perfectly (all zero)
    targets = np.array([0, 0, 0, 0])
    predictions = np.array([0, 0, 0, 0])
    attributes = np.array([1, 1, 0, 0])

    # Act
    result = DiscriminatoryBehavior.false_positive_parity(
        predictions, targets, attributes
    )

    # Assert: P(y_hat=1|y=0,a=1) = 0.0, P(y_hat=1|y=0,a=0) = 0.0. Diff = 0
    assert result == pytest.approx(0.0)


def test_accuracy_split():
    # Arrange: group 1 perfect, group 0 fully wrong
    predictions = np.array([1, 1, 0, 0])
    targets = np.array([1, 1, 1, 1])
    attributes = np.array([1, 1, 0, 0])

    # Act
    acc_true, acc_false = DiscriminatoryBehavior.accuracy(
        predictions, targets, attributes
    )

    # Assert
    assert acc_true == 100.0
    assert acc_false == 0.0
