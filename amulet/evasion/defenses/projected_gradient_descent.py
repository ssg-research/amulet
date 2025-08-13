"""Adversarial Training implementation"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

from .evasion_defense import EvasionDefense


class AdversarialTrainingPGD(EvasionDefense):
    """
    Implementation of Adversarial Training algorithm from the method from cleverhans:
    https://github.com/cleverhans-lab/cleverhans/blob/master/tutorials/torch/cifar10_tutorial.py.

    Reference:
        Towards Deep Learning Models Resistant to Adversarial Attacks
        Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu
        https://arxiv.org/abs/1706.06083.

    Attributes:
        model: :class:`~torch.nn.Module`
            The model on which to apply adversarial training.
        criterion: :class:`~torch.nn.Module`
            Loss function for adversarial training.
        optimizer: :class:`~torch.optim.Optimizer`
            Optimizer for adversarial training.
        train_loader: :class:`~torch.utils.data.DataLoader`
            Training data loader to adversarial training.
        device: str
            Device used to train model. Example: "cuda:0".
        epochs: int
            Determines number of iterations over training data.
        epsilon: int
            Controls the amount of perturbation on each image.
            Divided by 255. See: https://arxiv.org/abs/1412.6572.
        iterations: int
            Number of iterations for PGD generation.
        step_size: float
            Step size for each attack iteration.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        device: str,
        epochs: int = 5,
        epsilon: float = 0.1,
        iterations: int = 40,
        step_size: float = 0.01,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        super().__init__(model, criterion, optimizer, train_loader, device, epochs)
        self.epsilon = epsilon
        self.iterations = iterations
        self.step_size = step_size
        self.clip_min = clip_min
        self.clip_max = clip_max

    def train_robust(self) -> nn.Module:
        """
        Adversarially trains the model.

        Returns:
            Adversarially trained model of type :class:`torch.nn.Module'.
        """
        self.model.train()
        for epoch in range(self.epochs):
            correct = 0
            total = 0
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()

                _, predictions = torch.max(output, 1)
                correct += predictions.eq(y).sum().item()
                total += len(y)

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

                self.optimizer.zero_grad()
                output = self.model(x_pgd)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()

                _, predictions = torch.max(output, 1)
                correct += predictions.eq(y).sum().item()
                total += len(y)

            print(
                f"Train Epoch: {epoch} Loss: {loss.item():.6f} Acc: {correct/total*100:.2f}"  # type: ignore[reportPossiblyUnboundVariable]
            )

        return self.model
