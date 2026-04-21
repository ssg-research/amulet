"""Implementation of Model Inversion Attack from CCS 2015 by Fredrikson et al."""

import torch
import torch.nn as nn
import torch.utils.data

from .data_reconstruction_attack import DataReconstructionAttack


class FredriksonCCS2015(DataReconstructionAttack):
    """
    Model inversion attack reconstructing training data from confidence outputs.

    Reference:
        Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures (CCS 2015)
        Matt Fredrikson, Somesh Jha, Thomas Ristenpart
        https://rist.tech.cornell.edu/papers/mi-ccs.pdf

    Note: The model's output layer must include a Softmax. Add it manually if absent.

    Attributes:
        target_model: Target model whose training data is reconstructed.
        input_size: Shape of the model's input.
        output_size: Number of output classes.
        device: Device used for computation. Example: "cuda:0".
        alpha: Number of gradient descent iterations.
        beta: Early stopping window for cost comparison.
        gamma: Target cost threshold for early stopping.
        lamda: Gradient step size.
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

        self.target_model.to(self.device).eval()

    def __invert_cost(self, input_x: torch.Tensor) -> torch.Tensor:
        return 1 - self.target_model(input_x.requires_grad_(True))[0][self.target_label]

    def __model_invert(self) -> torch.Tensor:
        current_x = []
        cost_x = []
        current_x.append(
            torch.zeros(self.input_size, dtype=torch.float32).to(self.device)
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
        Reconstruct training data for each output class via gradient descent.

        Returns:
            List of reconstructed tensors, one per class.
        """
        reverse_data = []
        for i in range(self.output_size):
            self.target_label = i
            a = self.__model_invert()
            reverse_data.append(a)

        return reverse_data
