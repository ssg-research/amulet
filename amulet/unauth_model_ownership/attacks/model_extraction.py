"""Model Extraction implementation"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer


class ModelExtraction:
    """
    Implementation of algorithm to extract parameters of a model to
    obtain a "stolen" model. Code taken from:
    https://github.com/liuyugeng/ML-Doctor/blob/main/doctor/modsteal.py

    Reference:
        ML-Doctor: Holistic Risk Assessment of Inference Attacks Against Machine Learning Models,
        Yugeng Liu, Rui Wen, Xinlei He, Ahmed Salem, Zhikun Zhang, Michael Backes,
        Emiliano De Cristofaro, Mario Fritz, Yang Zhang,
        31st USENIX Security Symposium (USENIX Security 22)
        https://www.usenix.org/conference/usenixsecurity22/presentation/liu-yugeng


    Attributes:
        target_model: :class:`~torch.nn.Module`
            This model will be extracted.
        attack_model: :class:`~torch.nn.Module`
            The model trained by extracting target_model.
        optimizer: :class:`~torch.optim.Optimizer`
            Optimizer for training model.
        train_loader: :class:`~torch.utils.data.DataLoader`
            Dataloader for training model.
        device: str
            Device used to train model. Example: "cuda:0".
        epochs: int
            Determines number of iterations over training data.
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
        self.target_model = target_model
        self.attack_model = attack_model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = device
        self.epochs = epochs
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
        """
        Trains attack model by extracting the target model.

        Returns:
            Stolen model of type :class:`torch.nn.Module'.
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
                f"Epoch {epoch+1}/{self.epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%"
            )

        return self.attack_model
