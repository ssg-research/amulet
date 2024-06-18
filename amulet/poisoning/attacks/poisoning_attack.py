"""Poisoning Base Class"""

import torch
import torch.nn as nn


class PoisoningAttack:
    """
    Base Class for Poisoning attacks

    Attributes:
        poisoned_model: :class:`~torch.nn.Module`
            The model which will be trained on the poisoned dataset.
        optimizer: :class:~`torch.optim.Optimizer`
            The optimizer used to train the poisoned model.
        criterion: :class:~`torch.nn.Module`
            Loss function used to train the poisoned model.
        batch_size: int
            Batch used to train the poisoned model.
        device: str
            Device used for model inference. Example: "cuda:0".
        epochs: int
            Epochs used to train the poisoned model.
    """

    def __init__(
        self,
        poisoned_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        batch_size: int,
        device: str,
        epochs: int = 50,
    ):
        self.poisoned_model = poisoned_model
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.device = device
        self.epochs = epochs
