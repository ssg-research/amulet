"""Implementation of Model Inversion Attack from CCS 2015 by Fredrikson et al."""

import torch
import torch.nn as nn
import numpy as np
import torch.utils.data

from .data_reconstruction_attack import DataReconstructionAttack


class FredriksonCCS2015(DataReconstructionAttack):
    """
    Implementation of data reconstruction attack from the method from ML-Doctor Library:
    https://github.com/liuyugeng/ML-Doctor/blob/main/doctor/modinv.py

    Reference:
        Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures (CCS 2015)
        Matt Fredrikson, Somesh Jha, Thomas Ristenpart
        https://rist.tech.cornell.edu/papers/mi-ccs.pdf.

    -----------------------------NOTICE---------------------------
    If the model's output layer doesn't contain Softmax layer, please add it manually.
    And parameters will influence the quality of the reconstructed data significantly.
    --------------------------------------------------------------
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
        alpha: int
            Number of iterations
        beta, gamma, lambda: float
            Hyperparameters in paper
    """

    def __init__(
        self,
        target_model: nn.Module,
        input_size: tuple[int, ...],
        output_size: int,
        device: str,
        alpha: int,
        beta: int = 100,
        gamma: float = 0.001,
        lamda: float = 0.003,
    ):
        super().__init__(
            target_model,
            input_size,
            output_size,
            device,
        )

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lamda = lamda
        self.target_label = 1

        self.target_model.to(self.device).eval()

    def __invert_cost(self, input_x: torch.Tensor) -> torch.Tensor:
        return 1 - self.target_model(input_x.requires_grad_(True))[0][self.target_label]

    def __model_invert(self) -> torch.Tensor:
        current_x = []
        cost_x = []
        current_x.append(
            torch.Tensor(
                torch.from_numpy(np.zeros(self.input_size, dtype=np.float32))
            ).to(self.device)
        )

        for i in range(self.alpha):
            cost_x.append(self.__invert_cost(current_x[i]).to(self.device))
            cost_x[i].backward()
            current_x.append((current_x[i] - self.lamda * current_x[i].grad).data)
            if self.__invert_cost(current_x[i + 1]) <= self.gamma:
                print("Target cost value achieved")
                break
            elif i >= self.beta and self.__invert_cost(current_x[i + 1]) >= max(
                cost_x[self.beta : i + 1]
            ):
                print("Exceed beta")
                break

        i = cost_x.index(min(cost_x))
        return current_x[i]

    def attack(self) -> list[torch.Tensor]:
        """
        Outputs the reconstructed data of different classes

        Args:
            original_dataset: :class:~`torch.utils.data.DataLoader`
                The data used to train the target model, please make sure to set the batch size as 1.

        Returns:
            Reconstructed data for each class
        """
        reverse_data = []
        for i in range(self.output_size):
            self.target_label = i
            a = self.__model_invert()
            reverse_data.append(a)

        return reverse_data
