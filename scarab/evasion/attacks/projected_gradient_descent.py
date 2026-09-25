"""Evasion using Projected Gradient Descent implementation"""

import numpy as np
import torch
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset

from .evasion_attack import EvasionAttack


class EvasionPGD(EvasionAttack):
    """PGD-based evasion attack using the cleverhans library.

    Reference:
        Towards Deep Learning Models Resistant to Adversarial Attacks
        Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu
        https://arxiv.org/abs/1706.06083

    Attributes:
        model: The model to attack.
        test_loader: Input data that is perturbed to attack the model.
        device: Device used for model inference. Example: "cuda:0".
        batch_size: Batch size for the output DataLoader.
        epsilon: Perturbation budget.
        iterations: Number of PGD iterations.
        step_size: Step size for each attack iteration.
        clip_min: Lower bound to clamp perturbed inputs to.
        clip_max: Upper bound to clamp perturbed inputs to.
    """

    def __init__(
        self,
        model: Module,
        test_loader: DataLoader,
        device: str,
        batch_size: int,
        epsilon: float = 0.1,
        iterations: int = 40,
        step_size: float = 0.01,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        super().__init__(model, test_loader, device, batch_size)
        self.epsilon = epsilon
        self.iterations = iterations
        self.step_size = step_size
        self.clip_min = clip_min
        self.clip_max = clip_max

    def attack(self) -> DataLoader:
        """Run the PGD evasion attack and return perturbed inputs as a DataLoader.

        Returns:
            A DataLoader containing the adversarial examples.
        """
        self.model.eval()

        adv_input = []
        labels = []

        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            x_pgd = projected_gradient_descent(
                self.model,
                x,
                self.epsilon,
                self.step_size,
                self.iterations,
                np.inf,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
                sanity_checks=False,
            )

            adv_input.append(x_pgd.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        adv_input = np.concatenate(adv_input)
        labels = np.concatenate(labels)

        adversarial_test_set = TensorDataset(
            torch.from_numpy(adv_input).type(torch.float),
            torch.from_numpy(labels).type(torch.long),
        )
        adversarial_test_loader = DataLoader(
            dataset=adversarial_test_set, batch_size=self.batch_size, shuffle=False
        )

        return adversarial_test_loader
