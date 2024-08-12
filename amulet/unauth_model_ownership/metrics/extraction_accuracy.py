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
    Compares the attack model with the target model with respect to the accuracy,
    fidelity (agreement) and correct fidelity (agreement on correct predictions).

    Args:
        target_model: :class:`torch.nn.Module`
            Target model that was extracted.
        attack_model: :class:`torch.nn.Module`
            The surrogate model.
        data_loader: :class:`torch.utils.data.DataLoader`
            The testing data for model extraction.
        device: str
            Device on which to run the inference. Example "gpu:0"

    Returns:
        Dictionary containing the resulting values:
            'target_accuracy': Accuracy of the target model.
            'stolen_accuracy': Accuracy of the stolen (attack) model
            'fidelity': Fidelity between target and stolen model
            'correct_fidelity': Correctness conditioned on fidelity, i.e.
                                when models agree, how often are they also correct?
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
        "correct_fidelity": (both_correct / fidelity) * 100,
    }
