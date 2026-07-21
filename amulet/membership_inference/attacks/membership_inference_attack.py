"""Membership Inference Attack Base class"""

import random
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from ...utils import initialize_model


class MembershipInferenceAttack(ABC):
    """
    Base class for membership inference attacks.

    Attributes:
        shadow_architecture: Model architecture used for all shadow models.
        shadow_capacity: Size and complexity of the shadow model.
        train_set: The full dataset, a subset of which trains the target model.
        dataset: Name of the dataset.
        num_features: Number of features in the dataset.
        num_classes: Number of classes in the dataset.
        batch_size: Batch size used for training shadow models.
        pkeep: Proportion of training data to keep per shadow model.
        criterion: Loss function used to train shadow models.
        num_shadow: Number of shadow models to train.
        epochs: Number of training epochs for shadow models.
        device: Device used to train models. Example: "cuda:0".
        models_dir: Directory used to store shadow models.
        exp_id: Used as a random seed.
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
            shadow_model: The shadow model to be trained.
            train_loader: Training data for the shadow model.

        Returns:
            Trained shadow model.
        """
        device_type = self.device.split(":")[0]
        scaler = torch.amp.GradScaler(device_type, enabled=True)  # type: ignore[reportPrivateImportUsage]
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

                with torch.amp.autocast(device_type, enabled=True):  # type: ignore[reportPrivateImportUsage]
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
                f"Epoch {epoch + 1}/{self.epochs} — Loss: {epoch_loss:.4f} — Acc: {epoch_acc:.2f}%"
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
        keep = np.random.uniform(0, 1, size=(self.num_shadow, len(self.train_set)))  # type: ignore[reportArgumentType]
        order = keep.argsort(0)
        keep = order < int(self.pkeep * self.num_shadow)

        for shadow_id in range(self.num_shadow):
            filename = (
                self.models_dir / f"{self.dataset}_shadow_{shadow_id}_{self.exp_id}.pth"
            )
            if filename.exists():
                continue

            shadow_in_data = np.array(keep[shadow_id], dtype=bool).nonzero()[0]

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

            print(f"Saving shadow model #{shadow_id}")
            torch.save(
                {
                    "model": shadow_model.state_dict(),
                    "in_data": torch.as_tensor(shadow_in_data, dtype=torch.long),
                },
                filename,
            )

    def _load_shadow_model(self, shadow_id: int) -> tuple[nn.Module, np.ndarray]:
        """
        Load a trained shadow model and its training indices from disk.

        Args:
            shadow_id: Index of the shadow model to load.

        Returns:
            Tuple of (model, in_data) where model is eval-mode with gradients
            disabled, and in_data is the array of training-set indices this
            shadow model was trained on.

        Raises:
            FileNotFoundError: If the shadow model checkpoint does not exist.
        """
        path = self.models_dir / f"{self.dataset}_shadow_{shadow_id}_{self.exp_id}.pth"
        if not path.is_file():
            raise FileNotFoundError(f"Shadow model checkpoint not found: {path}")
        checkpoint = torch.load(path, weights_only=True)
        model = initialize_model(
            self.shadow_architecture,
            self.shadow_capacity,
            self.num_features,
            self.num_classes,
        )
        model.load_state_dict(checkpoint["model"])
        model.to(self.device)
        model.eval()
        model.requires_grad_(False)
        return model, checkpoint["in_data"].numpy()

    @abstractmethod
    def attack(self) -> dict[str, np.ndarray]:
        """Run the membership inference attack and return result arrays keyed by name."""
        pass
