"""Model Evasion Attack Base class"""

from abc import ABC, abstractmethod

import torch.nn as nn
from torch.utils.data import DataLoader


class EvasionAttack(ABC):
    """Base class for evasion attacks.

    Attributes:
        model: The model to attack.
        test_loader: Input data that is perturbed to attack the model.
        device: Device used for model inference. Example: "cuda:0".
        batch_size: Batch size for the output data loader.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str,
        batch_size: int,
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.batch_size = batch_size

    @abstractmethod
    def attack(self) -> DataLoader:
        """Run the attack and return the adversarial examples as a DataLoader."""
        pass
