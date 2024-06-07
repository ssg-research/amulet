"""Evasion Defense Base Class
"""

import copy
from typing import Optional

import torch

class EvasionDefense():
    """    
    Base class for Evasion Defense
    
    Attributes:
        model: :class:`~torch.nn.Module`
            The model on which to apply the defense.
        criterion: :class:`~torch.nn.Module`
            Loss function for the defense.
        optimizer: :class:`~torch.optim.Optimizer`
            Optimizer for the defense.
        train_loader: :class:`~torch.utils.data.DataLoader`
            Training data loader to the defense.
        device: str
            Device used to train model. Example: "cuda:0".
        epochs: int
            Determines number of iterations over training data.
    """
    def __init__(
            self,
            model: torch.nn.Module,
            criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            train_loader: torch.utils.data.DataLoader,
            device: str,
            epochs: Optional[int] = 5,
    ):
        self.model = copy.deepcopy(model)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = device
        self.epochs = epochs
