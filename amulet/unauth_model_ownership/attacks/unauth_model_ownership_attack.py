"""Unauthorized Model Ownership Attack Base class"""

from abc import ABC, abstractmethod

import torch.nn as nn


class UnauthModelOwnershipAttack(ABC):
    """
    Base class for unauthorized model ownership attacks.

    Unauthorized model ownership attacks attempt to replicate or steal
    a target model's functionality without authorization. Subclasses
    implement concrete model extraction or imitation strategies.

    Attributes:
        target_model: nn.Module
            The model being attacked (stolen from).
        device: str
            Device used for inference. Example: "cuda:0".
        epochs: int
            Number of training iterations for the attack model.
    """

    def __init__(
        self,
        target_model: nn.Module,
        device: str,
        epochs: int,
    ):
        self.target_model = target_model
        self.device = device
        self.epochs = epochs

    @abstractmethod
    def attack(self) -> nn.Module:
        pass
