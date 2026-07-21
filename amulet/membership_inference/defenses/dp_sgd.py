"""Differential Privacy implementation"""

from contextlib import nullcontext

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import DataLoader

from .membership_inference_defense import MembershipInferenceDefense


class DPSGD(MembershipInferenceDefense):
    """Differentially private SGD defense (DP-SGD) built on Opacus.

    Attributes:
        model: The model trained under differential privacy.
        criterion: Loss function for differentially private training.
        optimizer: Optimizer for differentially private training.
        train_loader: Training data loader for differentially private training.
        device: Device used to train model. Example: "cuda:0".
        delta: The target delta value for the differential privacy guarantee.
        max_per_sample_grad_norm: The norm to which the per-sample gradients are clipped.
        sigma: Noise multiplier.
        secure_rng: Whether to use secure RNG for trustworthy privacy guarantees. Comes at a performance cost.
        epochs: Determines number of iterations over training data.
        max_physical_batch_size: When set, each logical batch is split into physical
            micro-batches of at most this size (via Opacus' `BatchMemoryManager`) to bound
            peak per-sample-gradient memory. The logical batch size, and therefore the privacy
            accounting and expected utility, is unchanged. `None` (default) iterates the loader
            without splitting.
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
        max_physical_batch_size: int | None = None,
    ):
        super().__init__(model, criterion, optimizer, train_loader, device, epochs)
        self.privacy_engine = PrivacyEngine(secure_mode=secure_rng)
        self.delta = delta
        self.max_physical_batch_size = max_physical_batch_size
        self.model.train()
        self.model, self.optimizer, self.train_loader = (
            self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=sigma,
                max_grad_norm=max_per_sample_grad_norm,
            )
        )

    def train_private(self) -> nn.Module:
        """Train the model with differential privacy.

        Returns:
            The differentially private trained model.
        """

        self.model.train()

        for epoch in range(self.epochs):
            acc = 0
            total = 0
            last_loss: torch.Tensor | None = None
            # BatchMemoryManager splits each logical batch into physical micro-batches to
            # bound peak per-sample-gradient memory; nullcontext preserves today's behavior
            # exactly when no cap is set. Either way the per-step loop body is unchanged.
            if self.max_physical_batch_size is not None:
                batch_loader = BatchMemoryManager(
                    data_loader=self.train_loader,
                    max_physical_batch_size=self.max_physical_batch_size,
                    optimizer=self.optimizer,
                )
            else:
                batch_loader = nullcontext(self.train_loader)
            with batch_loader as loader:
                for batch_idx, (data, target) in enumerate(loader):
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    _, pred = torch.max(output, 1)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                    acc += pred.eq(target).sum().item()
                    total += len(target)
                    last_loss = loss
                    if batch_idx % 2000 == 0:
                        print(
                            f"Train Epoch: {epoch} Loss: {loss.item():.6f} Acc: {acc / total * 100:.2f}"
                        )
            if last_loss is not None:
                epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.delta)
                print(
                    f"Train Epoch: {epoch} Loss: {last_loss.item():.6f} (ε = {epsilon:.2f}, δ = {self.delta})"
                )
        print("Finished Training")

        return self.model
