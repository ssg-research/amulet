import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from amulet.unauth_model_ownership.metrics.extraction_accuracy import (
    evaluate_extraction,
)


class _FixedPredictionModel(nn.Module):
    """Deterministic model that emits one-hot logits at prescribed class indices."""

    def __init__(self, predictions):
        super().__init__()
        self.predictions = predictions

    def forward(self, x):
        batch_size = x.size(0)
        num_classes = 10
        logits = torch.zeros(batch_size, num_classes)
        for i in range(batch_size):
            logits[i, self.predictions[i]] = 1.0
        return logits


@pytest.fixture
def fixed_pred_model_factory():
    """Factory: build a model that returns the given top-class predictions."""
    return _FixedPredictionModel


@pytest.mark.parametrize(
    "target_preds, attack_preds, labels, expected",
    [
        # All correct: 100% across the board
        (
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            {
                "target_accuracy": 100.0,
                "stolen_accuracy": 100.0,
                "fidelity": 100.0,
                "correct_fidelity": 100.0,
            },
        ),
        # Fidelity 0: models never agree
        (
            torch.tensor([0, 0]),
            torch.tensor([1, 1]),
            torch.tensor([0, 0]),
            {
                "target_accuracy": 100.0,
                "stolen_accuracy": 0.0,
                "fidelity": 0.0,
                "correct_fidelity": 0.0,  # Guard check
            },
        ),
        # Partial overlap: target (50%), stolen (50%), fidelity (50%), both_correct (0%)
        # T: [0, 1], A: [1, 0], L: [0, 0]
        # Target correct on [0], Stolen correct on [1], they agree on nothing (fid 0)
        # Let's try: T: [0, 1], A: [0, 2], L: [0, 0]
        # Total=2. T correct: 1 (50%). S correct: 1 (50%). Fid (agree on 0): 1 (50%). Both correct (agree on 0 and L is 0): 1 (50%)
        (
            torch.tensor([0, 1]),
            torch.tensor([0, 2]),
            torch.tensor([0, 0]),
            {
                "target_accuracy": 50.0,
                "stolen_accuracy": 50.0,
                "fidelity": 50.0,
                "correct_fidelity": 100.0,  # both_correct (1) / fidelity (1)
            },
        ),
    ],
)
def test_evaluate_extraction_arithmetic(
    target_preds,
    attack_preds,
    labels,
    expected,
    cpu_device,
    fixed_pred_model_factory,
):
    target_model = fixed_pred_model_factory(target_preds)
    attack_model = fixed_pred_model_factory(attack_preds)
    dataset = TensorDataset(torch.randn(len(labels), 1), labels)
    loader = DataLoader(dataset, batch_size=len(labels))

    results = evaluate_extraction(target_model, attack_model, loader, cpu_device)

    for key, val in expected.items():
        assert results[key] == pytest.approx(val)


@pytest.mark.parametrize("target_seed, attack_seed", [(0, 1), (2, 3), (4, 5)])
def test_evaluate_extraction_outputs_within_bounds(
    target_seed,
    attack_seed,
    cpu_device,
    tiny_classifier_factory,
    tiny_loader,
    assert_within,
):
    target_model = tiny_classifier_factory(seed=target_seed, device=cpu_device)
    attack_model = tiny_classifier_factory(seed=attack_seed, device=cpu_device)

    results = evaluate_extraction(target_model, attack_model, tiny_loader, cpu_device)

    for key in (
        "target_accuracy",
        "stolen_accuracy",
        "fidelity",
        "correct_fidelity",
    ):
        assert_within(results[key], 0.0, 100.0)
