"""Adversarial Training implementation"""

import numpy as np
import torch
import torch.nn as nn
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .evasion_defense import EvasionDefense


class AdversarialTrainingPGD(EvasionDefense):
    """Adversarial training following the cleverhans CIFAR-10 tutorial.

    https://github.com/cleverhans-lab/cleverhans/blob/master/tutorials/torch/cifar10_tutorial.py

    Reference:
        Towards Deep Learning Models Resistant to Adversarial Attacks
        Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu
        https://arxiv.org/abs/1706.06083

    Attributes:
        model: The model on which to apply adversarial training.
        criterion: Loss function for adversarial training.
        optimizer: Optimizer for adversarial training.
        train_loader: Training data loader for adversarial training.
        device: Device used to train model. Example: "cuda:0".
        epochs: Number of iterations over the training data.
        epsilon: Perturbation budget applied directly in input space.
        iterations: Number of iterations for PGD generation.
        step_size: Step size for each attack iteration.
        clip_min: Lower bound to clamp perturbed inputs to.
        clip_max: Upper bound to clamp perturbed inputs to.
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
        """Adversarially train the model.

        Returns:
            The adversarially trained model.
        """
        self.model.train()
        for epoch in range(self.epochs):
            correct = 0
            total = 0
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                # Step 1: update on clean batch (joint clean+adversarial training variant).
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

                # Step 2: update on adversarial batch.
                self.optimizer.zero_grad()
                output = self.model(x_pgd)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()

                _, predictions = torch.max(output, 1)
                correct += predictions.eq(y).sum().item()
                total += len(y)

            print(
                f"Train Epoch: {epoch} Loss: {loss.item():.6f} Acc: {correct / total * 100:.2f}"  # type: ignore[reportPossiblyUnboundVariable]
            )

        return self.model
