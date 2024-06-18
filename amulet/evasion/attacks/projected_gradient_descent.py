"""Evasion using Projected Gradient Descent implementation"""

import torch
import numpy as np
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

from .evasion_attack import EvasionAttack


class EvasionPGD(EvasionAttack):
    """
    Implementation of evasion attack from the method from cleverhans:
    https://github.com/cleverhans-lab/cleverhans/blob/master/tutorials/torch/cifar10_tutorial.py.

    Reference:
        Towards Deep Learning Models Resistant to Adversarial Attacks
        Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu
        https://arxiv.org/abs/1706.06083.

    Attributes:
        model: :class:`~torch.nn.Module`
            The model on which to apply adversarial training.
        test_loader: :class:`~torch.utils.data.DataLoader`
            Input data that is perturbed to attack the model.
        device: str
            Device used for model inference. Example: "cuda:0".
        epsilon: int
            Controls the amount of perturbation on each image.
            Divided by 255. See: https://arxiv.org/abs/1412.6572.
        iterations: int
            Number of iterations for PGD generation.
        step_size: float
            Step size for each attack iteration.
        batch_size: int
            Batch size for output data loader.
    """

    def __init__(
        self,
        model: Module,
        test_loader: DataLoader,
        device: str,
        batch_size: int,
        epsilon: int = 16,
        iterations: int = 40,
        step_size: float = 0.02,
    ):
        super().__init__(model, test_loader, device, batch_size)
        self.epsilon = epsilon
        self.iterations = iterations
        self.step_size = step_size

    def run_evasion(self) -> DataLoader:
        """
        Runs the evasion attack on the model.

        Returns:
            Accuracy on adversarial examples as a percentage.
        """
        self.model.eval()

        adv_input = []
        labels = []

        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            x_pgd = projected_gradient_descent(
                self.model,
                x,
                self.epsilon / 255,
                self.step_size,
                self.iterations,
                np.inf,
            )

            adv_input.append(x_pgd.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        adv_input = np.concatenate(adv_input)
        labels = np.concatenate(labels)

        adversarial_test_set = TensorDataset(
            torch.from_numpy(adv_input).type(torch.float),
            torch.from_numpy(labels).type(torch.long),
        )
        advesrarial_test_loader = DataLoader(
            dataset=adversarial_test_set, batch_size=self.batch_size, shuffle=False
        )

        return advesrarial_test_loader
