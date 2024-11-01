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
    Base class for membership inference attacks

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
            Number of classes in dataset
        pkeep: float
            Proportion of training data to keep for shadow models (members vs non-members).
        criterion: :class:`~torch.nn.Module`
            The loss function used to train shadow models.
        num_shadow: int
            Number of shadow models to train.
        epochs: int
            Number of epochs used to train shadow models.
        device: str
            Device used to train model. Example: "cuda:0".
        models_dir: Path or str
            Directory used to store shadow models.
        experiment_id: int
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
        pkeep: float,
        criterion: nn.Module,
        num_shadow: int,
        epochs: int,
        device: str,
        models_dir: Path | str,
        experiment_id: int,
    ):
        self.shadow_architecture = shadow_architecture
        self.shadow_capacity = shadow_capacity
        self.dataset = dataset
        self.train_set = train_set
        self.epochs = epochs
        self.device = device
        self.num_features = num_features
        self.num_classes = num_classes
        self.criterion = criterion
        self.num_shadow = num_shadow
        self.pkeep = pkeep

        if isinstance(models_dir, str):
            models_dir = Path(models_dir)
        self.models_dir = models_dir

        torch.manual_seed(experiment_id)
        torch.cuda.manual_seed(experiment_id)
        torch.cuda.manual_seed_all(experiment_id)
        np.random.seed(experiment_id)
        torch.cuda.manual_seed_all(experiment_id)
        random.seed(experiment_id)

    def train_shadow_model(
        self, shadow_model: nn.Module, train_loader: DataLoader
    ) -> nn.Module:
        """
        Function used to train shadow models. Standard PyTorch training.

        Args:
            shadow_model: :class:~`torch.nn.Module`
                The shadow model to be trained.
            train_loader: :class:~`torch.utils.data.DataLoader`
                The training data for the shadow model.
        """
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        optimizer = torch.optim.SGD(
            shadow_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        for epoch in range(self.epochs):
            shadow_model.train()
            train_loss = 0
            correct = 0
            total = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # Train with amp
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = shadow_model(inputs)
                    loss = self.criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print(f"Epoch: {epoch}, Acc: {correct/total*100:.3f} ({correct}/{total})")
            scheduler.step()  # step cosine scheduling

        return shadow_model

    def prepare_shadow_models(self):
        """
        Splits the data into subsets for each shadow model,
        initializes and trains each shadow model. Saves the
        indices and the model in the specified directory.
        """
        # Generate random indices to get subset of data
        keep = np.random.uniform(0, 1, size=(self.num_shadow, len(self.train_set)))  # type: ignore[reportArgumentType]
        order = keep.argsort(0)
        keep = order < int(self.pkeep * self.num_shadow)

        for shadow_id in range(self.num_shadow):
            # Prepare data for shadow model
            shadow_in_data = np.array(keep[shadow_id], dtype=bool)
            shadow_in_data = shadow_in_data.nonzero()[0]
            train_subset = Subset(self.train_set, list(shadow_in_data))
            train_loader = DataLoader(
                train_subset, batch_size=128, shuffle=True, num_workers=4
            )

            print(f"Preparing shadow model #{shadow_id}")
            shadow_model = initialize_model(
                self.shadow_architecture,
                self.shadow_capacity,
                self.num_features,
                self.num_classes,
            )
            shadow_model.to(self.device)

            shadow_model = self.train_shadow_model(shadow_model, train_loader)

            print(f"Saving shadow model #{shadow_id}")
            state = {"model": shadow_model.state_dict(), "in_data": shadow_in_data}
            filename = self.models_dir / f"{self.dataset}_shadow_{shadow_id}.pth"
            torch.save(state, filename)


# TODO: Consider getting rid of this class entirely. Extra junk.
class InferenceModel(nn.Module):
    def __init__(
        self,
        shadow_id: int,
        dataset: str,
        num_features: int,
        num_classes: int,
        shadow_architecture: str,
        shadow_capacity: str,
        models_dir: Path | str,
    ):
        """
        Class used to load shadow models after they have been trained.
        Contains the indices of the data used to train the models
        as an attribute.

        Attributes:
            shadow_id: int
                The ID of the shadow model to be loaded.
            dataset:
                The dataset used to train the model.
            num_features:
                Number of features in the dataset.
            num_classes:
                Number of classes in the dataset.
            shadow_architecture:
                The architecture of the shadow model.
            shadow_capacity:
                The size and complexity of the shadow model.
            models_dir:
                The directory where shadow models were saved.
        """
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

        resume_checkpoint = self.models_dir / f"{self.dataset}_shadow_{shadow_id}.pth"

        assert os.path.isfile(resume_checkpoint), "Error: no checkpoint found!"
        checkpoint = torch.load(resume_checkpoint)

        self.model = initialize_model(
            self.shadow_architecture,
            self.shadow_capacity,
            self.num_features,
            self.num_classes,
        )
        self.model.load_state_dict(checkpoint["model"])

        self.in_data = checkpoint["in_data"]

        # no grad by default
        self.deactivate_grad()
        self.model.eval()

        self.is_in_model = False  # False for out_model

    def forward(self, x):
        return self.model(x)

    def deactivate_grad(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def activate_grad(self):
        for param in self.model.parameters():
            param.requires_grad = True

    # TODO: Fix parameter / return-type mismatch.
    def load_state_dict(self, checkpoint):  # type: ignore[reportIncompatibleMethodOverride]
        self.model.load_state_dict(checkpoint)
