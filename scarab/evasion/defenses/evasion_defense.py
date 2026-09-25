"""Evasion Defense Base Class"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EvasionDefense(ABC):
    """Base class for evasion defenses.

    Attributes:
        model: The model on which to apply the defense.
        criterion: Loss function for the defense.
        optimizer: Optimizer for the defense.
        train_loader: Training data loader for the defense.
        device: Device used to train model. Example: "cuda:0".
        epochs: Number of iterations over the training data.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        device: str,
        epochs: int = 5,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = device
        self.epochs = epochs

    @abstractmethod
    def train_robust(self) -> nn.Module:
        """Train the model robustly and return the defended model."""
        pass
