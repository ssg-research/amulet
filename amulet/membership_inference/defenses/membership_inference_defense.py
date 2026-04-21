"""Base class for Membership Inference defenses"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class MembershipInferenceDefense(ABC):
    """
    Base class for membership inference defenses.

    Attributes:
        model: torch.nn.Module
            The model on which to apply adversarial training.
        criterion: torch.nn.Module
            Loss function for adversarial training.
        optimizer: torch.optim.Optimizer
            Optimizer for adversarial training.
        train_loader: torch.utils.data.DataLoader
            Training data loader to adversarial training.
        device: str
            Device used to train model. Example: "cuda:0".
        epochs: int
            Determines number of iterations over training data.
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
        pass
