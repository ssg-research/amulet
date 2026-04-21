import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_extraction(
    target_model: nn.Module,
    attack_model: nn.Module,
    data_loader: DataLoader,
    device: str,
) -> dict:
    """
    Compare a stolen model against the target on accuracy, fidelity, and correct fidelity.

    Args:
        target_model: The original target model.
        attack_model: The surrogate (stolen) model.
        data_loader: Test data for evaluation.
        device: Device used for inference. Example: "cuda:0".

    Returns:
        Dictionary with keys "target_accuracy", "stolen_accuracy", "fidelity",
        and "correct_fidelity" (all as percentages).
    """
    target_model.eval()
    attack_model.eval()

    fidelity = 0
    target_correct = 0
    stolen_correct = 0
    both_correct = 0
    total = 0
    with torch.no_grad():
        for x, labels in data_loader:
            x, labels = x.to(device), labels.to(device)
            target_model_outputs = target_model(x)
            stolen_model_outputs = attack_model(x)
            _, target_model_predictions = torch.max(target_model_outputs.data, 1)
            _, stolen_model_predictions = torch.max(stolen_model_outputs.data, 1)

            total += target_model_predictions.size(0)
            fidelity += (
                (target_model_predictions == stolen_model_predictions).sum().item()
            )
            target_correct += (target_model_predictions == labels).sum().item()
            stolen_correct += (stolen_model_predictions == labels).sum().item()
            both_correct += (
                (
                    (stolen_model_predictions == labels)
                    & (target_model_predictions == labels)
                )
                .sum()
                .item()
            )

    return {
        "target_accuracy": (target_correct / total) * 100,
        "stolen_accuracy": (stolen_correct / total) * 100,
        "fidelity": (fidelity / total) * 100,
        "correct_fidelity": (both_correct / fidelity) * 100 if fidelity > 0 else 0.0,
    }
