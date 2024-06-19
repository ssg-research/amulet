"""Base class for Data Reconstruction Attacks"""

import torch.nn as nn


class DataReconstructionAttack:
    """
    Base class for Data Reconstruction Attacks
    Attributes::
        ------------------------
        target_model: :class:`~torch.nn.Module`
            Target model whose training dataset we reconstruct
        input_size: int
            Size of the model's input
        output_size: int
            Size of the model's output
        device: str
            Device on which to load the PyTorch tensors. Example: "cuda:0".
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
