"""Base class for poisoning defenses"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class PoisoningDefense:
    """
    Base class for Poisoning defenses.


    Attributes:
        model: :class:`~torch.nn.Module`
            The model to retrain after removing outliers.
        criterion: :class:`~torch.nn.Module`
            Loss function for training model.
        optimizer: :class:`~torch.optim.Optimizer`
            Optimizer for training model.
        train_loader: :class:`~torch.utils.data.DataLoader`
            Training data loader to train model.
        test_loader: :class:`~torch.utils.data.DataLoader`
            Test data loader to calculate Shapley values.
        device: str
            Device used to train model. Example: "cuda:0".
        epochs: int
            Determines number of iterations over training data.
        batch_size: int
            Batch size of training data.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str,
        epochs: int = 50,
        batch_size: int = 256,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
