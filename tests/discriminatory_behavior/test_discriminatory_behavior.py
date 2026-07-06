import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from amulet.discriminatory_behavior.metrics.discriminatory_behavior import (
    DiscriminatoryBehavior,
)


class _ClassFromInput(nn.Module):
    """Deterministic model: predicts the class encoded in the input's first feature.

    Defined at module scope so it's picklable; lets tests control predictions
    exactly via the data instead of depending on learned weights.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        idx = x[:, 0].long()
        logits = torch.zeros(x.shape[0], self.num_classes)
        logits[torch.arange(x.shape[0]), idx] = 1.0
        return logits


def _random_group_labels(rng: np.random.Generator, n: int) -> np.ndarray:
    """Build a binary label array of length n with at least one 0 and one 1."""
    arr = rng.integers(0, 2, size=n)
    if arr.min() == arr.max():
        arr[0] = 0
        arr[1] = 1
    return arr


def test_p_rule_divide_by_zero_guard():
    # y_z_0 (mean prediction for attribute=0) is 0
    predictions = np.array([1, 0, 0])
    attributes = np.array([1, 0, 0])

    result = DiscriminatoryBehavior.p_rule(predictions, attributes)

    assert result == 0.0


def test_p_rule_perfect_parity():
    # same positive rate for both groups
    predictions = np.array([1, 1, 0, 0])
    attributes = np.array([1, 0, 1, 0])

    result = DiscriminatoryBehavior.p_rule(predictions, attributes)

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
    result = DiscriminatoryBehavior.demographic_parity(predictions, attributes)

    assert result == pytest.approx(expected, abs=1e-7)


def test_true_positive_parity_identity():
    # predictions match targets perfectly
    targets = np.array([1, 1, 1, 1])
    predictions = np.array([1, 1, 1, 1])
    attributes = np.array([1, 1, 0, 0])

    result = DiscriminatoryBehavior.true_positive_parity(
        predictions, targets, attributes
    )

    # P(y_hat=1|y=1,a=1) = 1.0, P(y_hat=1|y=1,a=0) = 1.0. Diff = 0
    assert result == pytest.approx(0.0)


def test_false_positive_parity_identity():
    # predictions match targets perfectly (all zero)
    targets = np.array([0, 0, 0, 0])
    predictions = np.array([0, 0, 0, 0])
    attributes = np.array([1, 1, 0, 0])

    result = DiscriminatoryBehavior.false_positive_parity(
        predictions, targets, attributes
    )

    # P(y_hat=1|y=0,a=1) = 0.0, P(y_hat=1|y=0,a=0) = 0.0. Diff = 0
    assert result == pytest.approx(0.0)


# The identity cases above are degenerate: with predictions == targets and a
# single target class, the y-conditioning contributes nothing, so a metric that
# dropped the conditioning entirely (collapsing to demographic parity) would
# still score 0.0. The cases below use mixed predictions and both target classes
# so the y==1 / y==0 masks are load-bearing and pinned to hand-computed values.
_PARITY_ATTRS = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
_PARITY_TARGETS = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0])
_PARITY_PREDS = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1])


def test_true_positive_parity_pins_conditioned_rate():
    # a==0 group: TPR 1.0; a==1 group: TPR 0.0 -> parity 1.0. Dropping the y==1
    # mask would collapse this to demographic parity (0.1667 on these inputs).
    result = DiscriminatoryBehavior.true_positive_parity(
        _PARITY_PREDS, _PARITY_TARGETS, _PARITY_ATTRS
    )

    assert result == pytest.approx(1.0, abs=1e-6)


def test_false_positive_parity_pins_conditioned_rate():
    # a==1 group: FPR 1.0; a==0 group: FPR 0.333 -> parity 0.667. Dropping the
    # y==0 mask would collapse this to demographic parity (0.1667).
    result = DiscriminatoryBehavior.false_positive_parity(
        _PARITY_PREDS, _PARITY_TARGETS, _PARITY_ATTRS
    )

    assert result == pytest.approx(2 / 3, abs=1e-6)


def test_accuracy_split():
    # group 1 perfect, group 0 fully wrong
    predictions = np.array([1, 1, 0, 0])
    targets = np.array([1, 1, 1, 1])
    attributes = np.array([1, 1, 0, 0])

    acc_true, acc_false = DiscriminatoryBehavior.accuracy(
        predictions, targets, attributes
    )

    assert acc_true == 100.0
    assert acc_false == 0.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_accuracy_bounds(seed, assert_within):
    # random binary preds/targets, with each attribute group
    # guaranteed non-empty and containing both labels (accuracy_score requirement)
    rng = np.random.default_rng(seed)
    n_per_group = 10
    attributes = np.array([1] * n_per_group + [0] * n_per_group)
    targets = np.concatenate([
        _random_group_labels(rng, n_per_group),
        _random_group_labels(rng, n_per_group),
    ])
    predictions = rng.integers(0, 2, size=2 * n_per_group)

    acc_true, acc_false = DiscriminatoryBehavior.accuracy(
        predictions, targets, attributes
    )

    assert_within(acc_true, 0.0, 100.0)
    assert_within(acc_false, 0.0, 100.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_demographic_parity_bounds(seed, assert_within):
    rng = np.random.default_rng(seed)
    predictions = rng.integers(0, 2, size=20)
    attributes = rng.integers(0, 2, size=20)

    result = DiscriminatoryBehavior.demographic_parity(predictions, attributes)

    assert_within(result, 0.0, 1.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_true_positive_parity_bounds(seed, assert_within):
    rng = np.random.default_rng(seed)
    predictions = rng.integers(0, 2, size=20)
    targets = rng.integers(0, 2, size=20)
    attributes = rng.integers(0, 2, size=20)

    result = DiscriminatoryBehavior.true_positive_parity(
        predictions, targets, attributes
    )

    assert_within(result, 0.0, 1.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_false_positive_parity_bounds(seed, assert_within):
    rng = np.random.default_rng(seed)
    predictions = rng.integers(0, 2, size=20)
    targets = rng.integers(0, 2, size=20)
    attributes = rng.integers(0, 2, size=20)

    result = DiscriminatoryBehavior.false_positive_parity(
        predictions, targets, attributes
    )

    assert_within(result, 0.0, 1.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_p_rule_bounds(seed, assert_within):
    rng = np.random.default_rng(seed)
    predictions = rng.integers(0, 2, size=20)
    attributes = rng.integers(0, 2, size=20)

    result = DiscriminatoryBehavior.p_rule(predictions, attributes)

    assert_within(result, 0.0, 100.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_evaluate_subgroup_metrics_bounds(seed, assert_within):
    # random data/targets/sensitive attrs through a deterministic
    # class-from-input model, fed via a loader yielding (data, target, sensitive)
    torch.manual_seed(seed)
    n = 32
    n_attrs = 2
    data = torch.zeros(n, 4)
    data[:, 0] = torch.randint(0, 2, (n,)).float()
    targets = torch.randint(0, 2, (n,))
    sensitive = torch.randint(0, 2, (n, n_attrs))

    dataset = TensorDataset(data, targets, sensitive)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model = _ClassFromInput()
    evaluator = DiscriminatoryBehavior(model, loader, "cpu")

    metrics = evaluator.evaluate_subgroup_metrics()

    for i in range(n_attrs):
        assert_within(metrics[i]["acc_true"], 0.0, 100.0)
        assert_within(metrics[i]["acc_false"], 0.0, 100.0)
        assert_within(metrics[i]["demographic_parity"], 0.0, 1.0)
        assert_within(metrics[i]["true_positive_parity"], 0.0, 1.0)
        assert_within(metrics[i]["false_positive_parity"], 0.0, 1.0)
        assert_within(metrics[i]["p_rule"], 0.0, 100.0)
        assert_within(metrics[i]["equalized_odds"], 0.0, 2.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_adversary_auc_bounds(seed, assert_within):
    # identity main_model/discriminator; X float scores, Z binary
    # labels with both classes present per column (roc_auc_score requirement)
    torch.manual_seed(seed)
    n = 20
    n_attrs = 2
    X = torch.rand(n, n_attrs)
    Z = torch.randint(0, 2, (n, n_attrs))
    for col in range(n_attrs):
        if Z[:, col].min() == Z[:, col].max():
            Z[0, col] = 0
            Z[1, col] = 1
    y = torch.zeros(n)

    dataset = TensorDataset(X, y, Z)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    main_model = nn.Identity()
    discriminator = nn.Identity()

    aucs = DiscriminatoryBehavior.adversary_auc(
        discriminator, main_model, loader, "cpu"
    )

    for auc in aucs:
        assert_within(auc, 0.0, 1.0)


def test_subgroup_metrics_perfect_predictor_no_disparity():
    # first feature equals target, so _ClassFromInput predicts
    # targets exactly; single sensitive attribute, shape (4, 1)
    data = torch.zeros(4, 4)
    targets = torch.tensor([1, 0, 1, 0])
    data[:, 0] = targets.float()
    sensitive = torch.tensor([[1], [1], [0], [0]])

    dataset = TensorDataset(data, targets, sensitive)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = _ClassFromInput()
    evaluator = DiscriminatoryBehavior(model, loader, "cpu")

    metrics = evaluator.evaluate_subgroup_metrics()

    assert metrics[0]["acc_true"] == 100.0
    assert metrics[0]["acc_false"] == 100.0
    assert metrics[0]["demographic_parity"] == pytest.approx(0.0, abs=1e-7)
    assert metrics[0]["true_positive_parity"] == pytest.approx(0.0, abs=1e-7)
    assert metrics[0]["false_positive_parity"] == pytest.approx(0.0, abs=1e-7)
    assert metrics[0]["p_rule"] == pytest.approx(100.0)
    assert metrics[0]["equalized_odds"] == pytest.approx(0.0, abs=1e-7)


def test_subgroup_metrics_iterates_all_attributes():
    # two sensitive attributes, shape (N, 2)
    data = torch.zeros(8, 4)
    targets = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])
    data[:, 0] = targets.float()
    sensitive = torch.tensor([
        [1, 0],
        [1, 1],
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 0],
    ])

    dataset = TensorDataset(data, targets, sensitive)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = _ClassFromInput()
    evaluator = DiscriminatoryBehavior(model, loader, "cpu")

    metrics = evaluator.evaluate_subgroup_metrics()

    assert set(metrics.keys()) == {0, 1}
    expected_keys = {
        "acc_true",
        "acc_false",
        "demographic_parity",
        "true_positive_parity",
        "false_positive_parity",
        "p_rule",
        "equalized_odds",
    }
    for i in (0, 1):
        assert set(metrics[i].keys()) == expected_keys
        assert metrics[i]["equalized_odds"] == pytest.approx(
            metrics[i]["true_positive_parity"] + metrics[i]["false_positive_parity"]
        )


def test_adversary_auc_perfect_separation():
    # discriminator/main_model are identity, X equals Z exactly so
    # the discriminator's score perfectly separates the two classes
    main_model = nn.Identity()
    discriminator = nn.Identity()
    Z = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]])
    X = Z.float()
    y = torch.zeros(4)

    dataset = TensorDataset(X, y, Z)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    aucs = DiscriminatoryBehavior.adversary_auc(
        discriminator, main_model, loader, "cpu"
    )

    assert aucs == pytest.approx([1.0, 1.0])


def test_adversary_auc_uninformative_is_chance():
    # same Z, but X carries no information (constant 0.5 for all rows)
    main_model = nn.Identity()
    discriminator = nn.Identity()
    Z = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]])
    X = torch.full((4, 2), 0.5)
    y = torch.zeros(4)

    dataset = TensorDataset(X, y, Z)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    aucs = DiscriminatoryBehavior.adversary_auc(
        discriminator, main_model, loader, "cpu"
    )

    assert aucs == pytest.approx([0.5, 0.5])
