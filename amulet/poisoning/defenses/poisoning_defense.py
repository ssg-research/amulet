"""Base class for poisoning defenses."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class PoisoningDefense(ABC):
    """Base class for poisoning defenses.

    A poisoning defense produces a robust model via ``train_robust``: it cleans the
    (poisoned) training set — by removing outlier samples (``OutlierRemoval``) or purifying
    trigger content (``ONION``) — then retrains the victim on the cleaned data and returns
    it. Both defenses share that shape; only the cleaning mechanism differs.

    The retraining collaborators below are the standard, shared ingredients, and all are
    optional: a defense that only cleans data (e.g. ``ONION`` used purely to purify inputs
    at test time) can be constructed without them, while ``train_robust`` requires whatever
    it retrains with.

    Attributes:
        model: The victim model to retrain on the cleaned data.
        criterion: Loss function for retraining.
        optimizer: Optimizer for retraining.
        train_loader: Loader over the (poisoned) training data to clean and retrain on.
        test_loader: Loader used by cleaning steps that need held-out data (e.g. Shapley
            scoring in ``OutlierRemoval``).
        device: Device used to train the model. Example: "cuda:0".
        epochs: Number of iterations over the training data when retraining.
        batch_size: Batch size used to rebuild the cleaned training loader.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        criterion: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        train_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
        device: str = "cpu",
        epochs: int = 50,
        batch_size: int = 256,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size

    @abstractmethod
    def train_robust(self) -> nn.Module:
        """Clean the training set, retrain the victim on it, and return the robust model."""
