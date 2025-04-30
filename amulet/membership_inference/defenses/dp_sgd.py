"""Differential Privacy implementation"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from opacus import PrivacyEngine

from .membership_inference_defense import MembershipInferenceDefense


class DPSGD(MembershipInferenceDefense):
    """
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
        delt: float
            The target delta value for the differential privacy guarantee.
        max_per_sample_grad_norm: float,
            The norm to which the per-sample gradients are clipped.
        sigma:
            Noise multiplier
        secure_rng:
            Whether to use secure RNG for trustworthy privacy guarantees. Comes at a privacy cost.
        epochs: int
            Determines number of iterations over training data.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        device: str,
        delta: float,
        max_per_sample_grad_norm: float,
        sigma: float,
        secure_rng: bool = False,
        epochs: int = 5,
    ):
        super().__init__(model, criterion, optimizer, train_loader, device, epochs)
        self.privacy_engine = PrivacyEngine(secure_mode=secure_rng)
        self.delta = delta
        self.model.train()
        self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=sigma,
            max_grad_norm=max_per_sample_grad_norm,
        )

    def train_private(self) -> nn.Module:
        """
        Trains the model with differential privacy.

        Returns:
            Trained model of type :class:`torch.nn.Module'.
        """

        self.model.train()

        for epoch in range(self.epochs):
            acc = 0
            total = 0
            for batch_idx, (tuple) in enumerate(self.train_loader):
                data, target = tuple[0].to(self.device), tuple[1].to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                _, pred = torch.max(output, 1)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                acc += pred.eq(target).sum().item()
                total += len(target)
                if batch_idx % 2000 == 0:
                    print(
                        f"Train Epoch: {epoch} Loss: {loss.item():.6f} Acc: {acc/total*100:.2f}"
                    )
            epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.delta)
            print(
                f"Train Epoch: {epoch} Loss: {loss.item():.6f} (ε = {epsilon:.2f}, δ = {self.delta})"  # type: ignore[reportPossiblyUnboundVariable]
            )
        print("Finished Training")

        return self.model
