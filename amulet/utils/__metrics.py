"""
Utilities to evaluate models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
) -> float:
    """
    Calculates the classification accuracy of a model.

    Args:
        model: :class:`~nn.Module`
            The model to evaluate.
        data_loader: :class:'~torch.utils.data.DataLoader
            Input data to the model.
        device: str
            Device used for inference. Example: "cuda:0".

    Returns:
        The accuracy of the model.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for tuple in data_loader:
            x, y = tuple[0].to(device), tuple[1].to(device)
            outputs = model(x)
            _, predictions = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predictions == y).sum().item()

    return 100 * correct / total


def get_fidelity(
    model_1: nn.Module,
    model_2: nn.Module,
    data_loader: DataLoader,
    device: str,
) -> float:
    """
    Calculates the agreement in predictions (fidelity) of two models.

    Args:
        model: :class:`~nn.Module`
            One of the models to compare.
        model: :class:`~nn.Module`
            One of the models to compare.
        data_loader: :class:'~torch.utils.data.DataLoader
            Input data to the models.
        device: str
            Device used for inference. Example: "cuda:0".

    Returns:
        The agreement (fidelity) between two models.
    """
    model_1.eval()
    model_2.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            model_1_outputs = model_1(x)
            model_2_outputs = model_2(x)
            _, model_1_predictions = torch.max(model_1_outputs.data, 1)
            _, model_2_predictions = torch.max(model_2_outputs.data, 1)

            total += model_1_predictions.size(0)

            correct += (model_1_predictions == model_2_predictions).sum().item()

    return 100 * correct / total
