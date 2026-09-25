"""Base class for Membership Inference defenses"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class MembershipInferenceDefense(ABC):
    """Base class for membership inference defenses.

    Attributes:
        model: The model to train with the membership inference defense.
        criterion: Loss function for defense training.
        optimizer: Optimizer for defense training.
        train_loader: Training data loader for defense training.
        device: Device used to train model. Example: "cuda:0".
        epochs: Determines number of iterations over training data.
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
    def train_private(self) -> nn.Module:
        """Train the model with the membership inference defense and return it."""
        pass
