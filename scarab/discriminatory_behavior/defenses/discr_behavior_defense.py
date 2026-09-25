"""Base class for Discriminatory Behavior defenses"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class DiscriminatoryBehaviorDefense(ABC):
    """Base class for Discriminatory Behavior defenses.

    Attributes:
        model: The model on which to apply adversarial training.
        criterion: Loss function for adversarial training.
        optimizer: Optimizer for adversarial training.
        train_loader: Training data loader for adversarial training.
        test_loader: Testing data loader for adversarial training.
        device: Device used to train model. Example: "cuda:0".
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    @abstractmethod
    def train_fair(self) -> nn.Module:
        """Train the model to mitigate discriminatory behavior.

        Returns:
            The debiased model.
        """
