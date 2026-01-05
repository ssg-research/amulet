"""Membership Inference Attack Base class"""

import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from ...utils import initialize_model


class MembershipInferenceAttack:
    """
    Base class for membership inference attacks.

    Attributes:
        shadow_architecture: str
            The model architecture used for all shadow models.
        shadow_capacity: str
            Size and complexity of the shadow model.
        train_set: :class:`~torch.utils.data.Dataset`
            The full dataset, a subset of which is used to train the target model.
        dataset: str
            The name of the dataset.
        num_features: int
            Number of features in dataset.
        num_classes: int
            Number of classes in dataset.
        batch_size: int
            Batch size used for training shadow models.
        pkeep: float
            Proportion of training data to keep for shadow models (members vs non-members).
        criterion: :class:`~torch.nn.Module`
            The loss function used to train shadow models.
        num_shadow: int
            Number of shadow models to train.
        epochs: int
            Number of epochs used to train shadow models.
        device: str
            Device used to train models. Example: "cuda:0".
        models_dir: Path or str
            Directory used to store shadow models.
        exp_id: int
            Used as a random seed.
    """

    def __init__(
        self,
        shadow_architecture: str,
        shadow_capacity: str,
        train_set: Dataset,
        dataset: str,
        num_features: int,
        num_classes: int,
        batch_size: int,
        pkeep: float,
        criterion: nn.Module,
        num_shadow: int,
        epochs: int,
        device: str,
        models_dir: Path | str,
        exp_id: int,
    ):
        self.shadow_architecture = shadow_architecture
        self.shadow_capacity = shadow_capacity
        self.dataset = dataset
        self.train_set = train_set
        self.epochs = epochs
        self.device = device
        self.num_features = num_features
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.pkeep = pkeep
        self.criterion = criterion
        self.num_shadow = num_shadow
        self.exp_id = exp_id

        if isinstance(models_dir, str):
            models_dir = Path(models_dir)
        self.models_dir = models_dir

        # Set random seeds for reproducibility
        torch.manual_seed(exp_id)
        torch.cuda.manual_seed(exp_id)
        torch.cuda.manual_seed_all(exp_id)
        np.random.seed(exp_id)
        random.seed(exp_id)

    def train_shadow_model(
        self, shadow_model: nn.Module, train_loader: DataLoader
    ) -> nn.Module:
        """
        Train a single shadow model.

        Args:
            shadow_model: :class:`torch.nn.Module`
                The shadow model to be trained.
            train_loader: :class:`torch.utils.data.DataLoader`
                DataLoader containing training data for the shadow model.

        Returns:
            nn.Module: The trained shadow model.
        """
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        optimizer = torch.optim.SGD(
            shadow_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        for epoch in range(self.epochs):
            shadow_model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                with torch.cuda.amp.autocast(enabled=True):
                    outputs = shadow_model(inputs)
                    loss = self.criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            epoch_loss = train_loss / total
            epoch_acc = 100.0 * correct / total
            print(
                f"Epoch {epoch+1}/{self.epochs} — Loss: {epoch_loss:.4f} — Acc: {epoch_acc:.2f}%"
            )
            scheduler.step()

        return shadow_model

    def prepare_shadow_models(self):
        """
        Prepares and trains all shadow models.

        Splits the dataset into subsets for each shadow model,
        initializes and trains them, then saves each model
        and the indices used for training.
        """
        # Randomly assign data indices to shadow models to get subset
        keep = np.random.uniform(0, 1, size=(self.num_shadow, len(self.train_set)))  # type: ignore
        order = keep.argsort(0)
        keep = order < int(self.pkeep * self.num_shadow)

        for shadow_id in range(self.num_shadow):
            filename = (
                self.models_dir / f"{self.dataset}_shadow_{shadow_id}_{self.exp_id}.pth"
            )
            if filename.exists():
                continue
            # Select indices for this shadow model
            shadow_in_data = np.array(keep[shadow_id], dtype=bool)
            shadow_in_data = shadow_in_data.nonzero()[0]

            train_subset = Subset(self.train_set, list(shadow_in_data))
            train_loader = DataLoader(
                train_subset, batch_size=self.batch_size, shuffle=True, num_workers=4
            )

            print(
                f"Preparing shadow model #{shadow_id} with {len(shadow_in_data)} samples"
            )
            shadow_model = initialize_model(
                self.shadow_architecture,
                self.shadow_capacity,
                self.num_features,
                self.num_classes,
            )
            shadow_model.to(self.device)

            shadow_model = self.train_shadow_model(shadow_model, train_loader)

            # Save model state dict and training indices
            print(f"Saving shadow model #{shadow_id}")
            state = {"model": shadow_model.state_dict(), "in_data": shadow_in_data}
            torch.save(state, filename)


class InferenceModel(nn.Module):
    """
    Wrapper to load a shadow model after training.

    Attributes:
        shadow_id: int
            ID of the shadow model.
        dataset: str
            Dataset name.
        num_features: int
            Number of input features.
        num_classes: int
            Number of output classes.
        shadow_architecture: str
            Model architecture.
        shadow_capacity: str
            Model capacity descriptor.
        models_dir: Path or str
            Directory where models are stored.
        in_data: np.ndarray
            Indices of training data for this shadow model.
        model: nn.Module
            Loaded shadow model.
    """

    def __init__(
        self,
        shadow_id: int,
        dataset: str,
        num_features: int,
        num_classes: int,
        shadow_architecture: str,
        shadow_capacity: str,
        models_dir: Path | str,
        exp_id: int,
    ):
        super().__init__()

        self.shadow_id = shadow_id
        self.dataset = dataset
        self.num_features = num_features
        self.num_classes = num_classes
        self.shadow_architecture = shadow_architecture
        self.shadow_capacity = shadow_capacity

        if isinstance(models_dir, str):
            models_dir = Path(models_dir)
        self.models_dir = models_dir

        resume_checkpoint = (
            self.models_dir / f"{self.dataset}_shadow_{shadow_id}_{exp_id}.pth"
        )
        assert os.path.isfile(
            resume_checkpoint
        ), f"Checkpoint not found at {resume_checkpoint}"
        checkpoint = torch.load(resume_checkpoint)

        self.model = initialize_model(
            self.shadow_architecture,
            self.shadow_capacity,
            self.num_features,
            self.num_classes,
        )
        self.model.load_state_dict(checkpoint["model"])

        self.in_data = checkpoint["in_data"]

        self.deactivate_grad()
        self.model.eval()

        self.is_in_model = False  # Flag indicating whether this model represents member (True) or non-member (False)

    def forward(self, x):
        return self.model(x)

    def deactivate_grad(self):
        """Disable gradients for all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def activate_grad(self):
        """Enable gradients for all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = True

    def load_state_dict(self, checkpoint):  # type: ignore[reportIncompatibleMethodOverride]
        """Load state dict into the underlying model."""
        self.model.load_state_dict(checkpoint)
