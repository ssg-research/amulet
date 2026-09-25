"""Model extraction attack that distills a target model into a surrogate."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .unauth_model_ownership_attack import UnauthModelOwnershipAttack


class ModelExtraction(UnauthModelOwnershipAttack):
    """Extract a target model into a "stolen" surrogate by distillation.

    Code adapted from
    https://github.com/liuyugeng/ML-Doctor/blob/main/doctor/modsteal.py.

    Reference:
        ML-Doctor: Holistic Risk Assessment of Inference Attacks Against Machine Learning Models,
        Yugeng Liu, Rui Wen, Xinlei He, Ahmed Salem, Zhikun Zhang, Michael Backes,
        Emiliano De Cristofaro, Mario Fritz, Yang Zhang,
        31st USENIX Security Symposium (USENIX Security 22)
        https://www.usenix.org/conference/usenixsecurity22/presentation/liu-yugeng

    Attributes:
        target_model: The model to extract (steal from).
        attack_model: The surrogate model trained to imitate the target.
        optimizer: Optimizer for training the surrogate.
        train_loader: Data loader used to query the target and train the surrogate.
        device: Device used to train the model. Example: "cuda:0".
        epochs: Number of iterations over the training data.
        loss_type: Distillation loss matching the surrogate to the target. One of
            "mse" (regress logits), "kl" (match softmax distributions), or "ce"
            (train on the target's hard labels).

    Raises:
        ValueError: If `loss_type` is not one of "mse", "kl", or "ce".
    """

    def __init__(
        self,
        target_model: nn.Module,
        attack_model: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        device: str,
        epochs: int = 50,
        loss_type: str = "mse",  # "kl", "ce" are other options
    ):
        super().__init__(target_model, device, epochs)
        self.attack_model = attack_model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.loss_type = loss_type.lower()

        if self.loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif self.loss_type == "kl":
            self.criterion = nn.KLDivLoss(reduction="batchmean")
        elif self.loss_type == "ce":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

    def attack(self) -> nn.Module:
        """Train the surrogate model by extracting the target model.

        Returns:
            The stolen (surrogate) model.
        """
        self.attack_model.train()
        self.target_model.eval()

        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                # Forward and loss computation
                if self.loss_type == "mse":
                    with torch.no_grad():
                        target_logits = self.target_model(x)
                    output_logits = self.attack_model(x)
                    loss = self.criterion(output_logits, target_logits)
                    preds = output_logits.argmax(dim=1)

                elif self.loss_type == "kl":
                    with torch.no_grad():
                        target_probs = F.softmax(self.target_model(x), dim=1)
                    output_log_probs = F.log_softmax(self.attack_model(x), dim=1)
                    loss = self.criterion(output_log_probs, target_probs)
                    preds = output_log_probs.exp().argmax(
                        dim=1
                    )  # convert log_probs back to probs

                elif self.loss_type == "ce":
                    with torch.no_grad():
                        target_labels = self.target_model(x).argmax(dim=1)
                    logits = self.attack_model(x)
                    loss = self.criterion(logits, target_labels)
                    preds = logits.argmax(dim=1)
                else:
                    raise ValueError(f"Unsupported loss_type: {self.loss_type}")

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Metrics
                running_loss += loss.item() * x.size(0)
                total += y.size(0)
                correct += preds.eq(y).sum().item()

            avg_loss = running_loss / total
            acc = 100.0 * correct / total

            print(
                f"Epoch {epoch + 1}/{self.epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%"
            )

        return self.attack_model
