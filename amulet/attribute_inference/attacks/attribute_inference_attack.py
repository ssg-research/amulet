"""Attribute Inference Attack Base class"""

from abc import ABC, abstractmethod

import numpy as np
import torch.nn as nn


class AttributeInferenceAttack(ABC):
    """
    Base class for attribute inference attacks

    Attributes:
        target_model: torch.nn.Module
            This model will be extracted.
        x_train_adv: numpy.ndarray
            input features for training adversary' attack model
        x_test: numpy.ndarray
            input features for testing adversary' attack model
        z_train_adv: numpy.ndarray
            sensitive attributes for training adversary' attack model
        device: str
            Device used to train model. Example: "cuda:0".
    """

    def __init__(
        self,
        target_model: nn.Module,
        x_train_adv: np.ndarray,
        x_test: np.ndarray,
        z_train_adv: np.ndarray,
        device: str,
    ):
        self.target_model = target_model
        self.x_train_adv = x_train_adv
        self.z_train_adv = z_train_adv
        self.x_test = x_test
        self.device = device

    @abstractmethod
    def attack(self) -> dict[int, dict[str, np.ndarray]]:
        pass
