"""Base class for Discriminatory Behavior defenses"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class DicriminatoryBehaviorDefense:
    """
    Base class for Discriminatory Behavior defenses

    Attributes:
        model: :class:`~torch.nn.Module`
            The model on which to apply adversarial training.
        criterion: :class:`~torch.nn.Module`
            Loss function for adversarial training.
        optimizer: :class:`~torch.optim.Optimizer`
            Optimizer for adversarial training.
        train_loader: :class:`~torch.utils.data.DataLoader`
            Training data loader to adversarial training.
        test_loader: :class:`~torch.utils.data.DataLoader`
            Testing data loader to adversarial training.
        lambdas: :class:`torch.Tensor`
            Hyperparameters for fairness objective function
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
        test_loader: DataLoader,
        device: str,
    ):
        self.model = model
        self.model_criterion = criterion
        self.model_optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
