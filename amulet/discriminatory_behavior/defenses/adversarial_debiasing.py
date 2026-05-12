"""Adversarial Debiasing Implementation"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .discr_behavior_defense import DiscriminatoryBehaviorDefense


class AdversaryModel(nn.Module):
    """
    Model used as a discriminator to identify the sensitive
    attribute given the output of the model.
    Attributes:
        n_sensitive_attrs: int
            Number of sensitive attributes in the dataset.
        n_classes: int
            Number of classes in the dataset.
    """

    def __init__(self, n_sensitive_attrs: int = 2, n_classes: int = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_classes, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_sensitive_attrs),
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))


class AdversarialDebiasing(DiscriminatoryBehaviorDefense):
    """
    Adversarial debiasing defense that jointly trains a classifier and a discriminator.

    Reference:
        Learning to Pivot with Adversarial Networks
        Gilles Louppe, Michael Kagan, Kyle Cranmer

    Attributes:
        model: The main classifier to debias.
        criterion: Loss function for the main classifier.
        optimizer: Optimizer for the main classifier.
        train_loader: Training data loader yielding (X, y, Z) batches.
        test_loader: Test data loader yielding (X, y, Z) batches.
        n_sensitive_attrs: Number of sensitive attributes in the dataset.
        n_classes: Number of output classes.
        lambdas: Per-attribute fairness penalty weights.
        device: Device used to train model. Example: "cuda:0".
        epochs: Number of training epochs.
        pretrain_adversary_epochs: Number of epochs to pretrain the adversary before joint training.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        test_loader: DataLoader,
        n_sensitive_attrs: int,
        n_classes: int,
        lambdas: torch.Tensor,
        device: str,
        epochs: int = 5,
        pretrain_adversary_epochs: int = 5,
    ):
        super().__init__(model, criterion, optimizer, train_loader, test_loader, device)

        self.lambdas = lambdas.to(device)
        self.epochs = epochs
        self.n_sensitive_attrs = n_sensitive_attrs

        self.pretrain_adversary_epochs = pretrain_adversary_epochs
        self.discmodel = AdversaryModel(n_sensitive_attrs, n_classes).to(self.device)
        self.adv_criterion = nn.BCELoss(reduction="none")
        self.adv_optimizer = torch.optim.Adam(self.discmodel.parameters())
        self.discmodel = self.__pretrain_adversary()

    def train_fair(self) -> nn.Module:
        """
        Train the classifier with adversarial debiasing.

        Returns:
            Debiased model.
        """
        print("Training Model with Adversarial Debiasing")
        self.model.train()
        for _ in range(self.epochs):
            for x, y, z in self.train_loader:
                x, y, z = x.to(self.device), y.to(self.device), z.to(self.device)

                # Update adversary; detach so gradients don't flow into the main model.
                self.discmodel.zero_grad()
                p_z = self.discmodel(self.model(x).detach())
                loss_adv = (self.adv_criterion(p_z, z) * self.lambdas).mean()
                loss_adv.backward()
                self.adv_optimizer.step()

                # Update main model to minimise task loss and maximise adversary loss.
                self.model.zero_grad()
                p_y = self.model(x)
                model_loss = (
                    self.criterion(p_y, y)
                    - (self.adv_criterion(self.discmodel(p_y), z) * self.lambdas).mean()
                )
                model_loss.backward()
                self.optimizer.step()

        return self.model

    def __pretrain_adversary(self):
        print("Pretraining Adversary Model")
        for _ in range(self.pretrain_adversary_epochs):
            for x, _, z in self.train_loader:
                x, z = x.to(self.device), z.to(self.device)
                p_y = self.model(x).detach()
                self.discmodel.zero_grad()
                p_z = self.discmodel(p_y)
                loss = (self.adv_criterion(p_z, z) * self.lambdas).mean()
                loss.backward()
                self.adv_optimizer.step()

        return self.discmodel
