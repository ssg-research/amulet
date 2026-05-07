"""
Utilities to train and provide white-box access to models.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..models import AmuletModel


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
    Train a classifier.

    Args:
        model: Model to be trained.
        data_loader: Data used to train the model.
        criterion: Loss function for training model.
        optimizer: Optimizer for training model.
        epochs: Number of iterations over training data.
        device: Device used to train model. Example: "cuda:0".
        scheduler: LR scheduler.
        early_stopping_patience: Stop if no accuracy improvement after this many epochs.

    Returns:
        Trained model.
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


def load_or_train(
    path: Path,
    init_fn: Callable[[], nn.Module],
    train_fn: Callable[[nn.Module], nn.Module],
    log: logging.Logger | None = None,
    description: str = "model",
) -> nn.Module:
    """
    Load a model checkpoint if it exists, otherwise train and save one.

    Collapses the standard `if checkpoint.exists(): load else train+save` block
    that every example pipeline reimplements. The caller supplies two closures:
    `init_fn` returns a freshly initialised model, and `train_fn` trains it.
    On a cache miss, `init_fn` is invoked, the result is trained via `train_fn`,
    and its state_dict is saved to `path` (parent directories created as needed).
    On a cache hit, `init_fn` is invoked and its state_dict is loaded from `path`.

    Args:
        path: Checkpoint file path. Parent directories are created on save.
        init_fn: Zero-argument callable returning a fresh `nn.Module` already on
            the target device.
        train_fn: Callable that takes the freshly initialised model and returns
            the trained model. Responsible for optimizer, loaders, and epochs.
        log: Optional logger for progress messages.
        description: Short noun phrase describing the model, used in log lines
            (e.g. "target model", "defended model", "shadow model").

    Returns:
        The loaded or newly trained model.
    """
    if path.exists():
        if log:
            log.info("Loading %s from %s", description, path)
        model = init_fn()
        model.load_state_dict(torch.load(path, weights_only=True))
        return model

    if log:
        log.info("Training %s", description)
    model = init_fn()
    model = train_fn(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    if log:
        log.info("Saved %s to %s", description, path)
    return model


def get_predictions_numpy(
    input_data: np.ndarray, model: nn.Module, batch_size: int, device: str
) -> np.ndarray:
    """
    Get predictions from a model for numpy input and return as a numpy array.

    Args:
        input_data: The input for the model.
        model: Model to get predictions from.
        batch_size: Batch size for the data loader.
        device: Device used for computation.

    Returns:
        Model predictions as a numpy array.
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
    model: AmuletModel, data_loader: DataLoader, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get intermediate layer outputs, labels, and inputs from a model.

    Args:
        model: Amulet model to get intermediate outputs from.
        data_loader: Input data to the model.
        device: Device used for inference. Example: "cuda:0".

    Returns:
        Tuple of (intermediate features, labels, inputs) as numpy arrays.
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

    features = np.concatenate(features, axis=0)
    targets = np.concatenate(targets, axis=0)
    inputs = np.concatenate(inputs, axis=0)

    return features, targets, inputs
