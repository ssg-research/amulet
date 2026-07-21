"""Base class for Data Reconstruction Attacks."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class DataReconstructionAttack(ABC):
    """Base class for data reconstruction attacks.

    Attributes:
        target_model: Target model whose training dataset is reconstructed.
        input_size: Shape of the model's input.
        output_size: Size of the model's output.
        device: Device on which to load the PyTorch tensors. Example: "cuda:0".
    """

    def __init__(
        self,
        target_model: nn.Module,
        input_size: tuple[int, ...],
        output_size: int,
        device: str,
    ):
        self.target_model = target_model
        self.input_size = input_size
        self.output_size = output_size
        self.device = device

    @abstractmethod
    def attack(self) -> list[torch.Tensor]:
        """Reconstruct training data for each output class.

        Returns:
            List of reconstructed tensors, one per class.
        """
        pass
