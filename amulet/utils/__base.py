"""
Utilities to train and provide white-box access to models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def train_classifier(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    early_stopping_patience: int = 25,
) -> nn.Module:
    """
    Trains a classifier.

    Args:
        model: :class:`~torch.nn.Module`
            Model to be trained.
        data_loader: :class:'~torch.utils.data.DataLoader`
            Data used to train the model.
        criterion: :class:`~torch.nn.Module`
            Loss function for training model.
        optimizer: :class:`~torch.optim.Optimizer`
            Optimizer for training model.
        epochs: int
            Determines number of iterations over training data.
        device: str
            Device used to train model. Example: "cuda:0".
        scheduler: :class:`~torch.optim.lr_scheduler._LRScheduler`
            LR scheduler.
        early_stopping_patience: int
            Stop if no accuracy improvement after this many epochs.

    Returns:
        Trained model of type :class:`~torch.nn.Module`.
    """
    model.train()
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        correct = 0
        total = 0
        running_loss = 0.0

        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            _, predictions = torch.max(output, 1)
            correct += predictions.eq(y).sum().item()
            total += y.size(0)
            running_loss += loss.item() * y.size(0)

        acc = correct / total * 100
        avg_loss = running_loss / total

        if scheduler:
            scheduler.step()

        print(
            f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Early stopping check
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (best acc: {best_acc:.2f}%)")
                break

    print("Finished training")
    return model


def get_predictions_numpy(
    input_data: np.ndarray, model: nn.Module, batch_size: int, device: str
) -> np.ndarray:
    """
    Helper function to get predictions from a model from a numpy array and return them as a numpy array.

    Args:
        input_data: :class:`~np.ndarray`
            The input for the model.
        model: :class:`nn.Module`
            Get predictions from this model.
        batch_size: int
            Batch size for the data loader.
        device: str
            Device used for computation.

    Returns:
        Predictions of type :class:`~np.ndarray`.
    """
    dataloader = DataLoader(
        dataset=TensorDataset(torch.from_numpy(input_data).type(torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    predictions_list = []
    with torch.no_grad():
        for (x,) in dataloader:
            x = x.to(device)
            predictions = model(x).cpu().numpy()
            predictions_list.append(predictions)

    all_predictions = np.concatenate(predictions_list, axis=0)

    return all_predictions


def get_intermediate_features(
    model: nn.Module, data_loader: DataLoader, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gets the intermediate layer output of a model.

    Args:
        model: :class:`~torch.nn.Module`
            Model to get intermediate layer outputs from.
            Assumes model has a .get_hidden() function.
            See sample models in src.models.
        data_loader: :class:'~torch.utils.data.DataLoader
            Input data to the model.
        device: str
            Device used for inference. Example: "cuda:0".

    Returns:
        A tuple containing:
            The intermediate layer output of the model of type :class:`~torch.Tensor`.

            The labels of type :class:`~torch.Tensor`.

            The inputs of type :class:`~torch.Tensor`.
    """
    model.eval()
    features = []
    targets = []
    inputs = []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            feature = model.get_hidden(x).data.cpu().numpy()

            features.append(feature)
            targets.append(y.data.cpu().numpy())
            inputs.append(x.data.cpu().numpy())

    features = np.concatenate(np.array(features, dtype=object))
    targets = np.concatenate(np.array(targets, dtype=object))
    inputs = np.concatenate(np.array(inputs, dtype=object))

    return features, targets, inputs
