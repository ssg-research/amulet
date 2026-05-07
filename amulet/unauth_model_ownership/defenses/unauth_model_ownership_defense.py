"""Unauthorized Model Ownership Defense Base classes"""

from abc import ABC, abstractmethod

import torch.nn as nn


class WatermarkDefense(ABC):
    """
    Base class for watermarking-based ownership defenses.

    Embeds a secret trigger set into the model during or after training so
    that the owner can later verify ownership by querying the model on
    those triggers.

    Attributes:
        target_model: nn.Module
            The model to protect.
        device: str
            Device used for inference. Example: "cuda:0".
    """

    def __init__(self, target_model: nn.Module, device: str):
        self.target_model = target_model
        self.device = device

    @abstractmethod
    def watermark(self) -> nn.Module:
        pass


class FingerprintDefense(ABC):
    """
    Base class for fingerprinting-based ownership defenses.

    Identifies whether a suspect model was stolen from the target model
    by analyzing the target's training data distribution.

    Attributes:
        target_model: nn.Module
            The model to protect.
        device: str
            Device used for inference. Example: "cuda:0".
    """

    def __init__(self, target_model: nn.Module, device: str):
        self.target_model = target_model
        self.device = device

    @abstractmethod
    def fingerprint(self) -> dict[str, dict[str, float]]:
        pass
