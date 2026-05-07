"""Poisoning Attack Base class"""

from abc import ABC, abstractmethod

from torch.utils.data import TensorDataset


class PoisoningAttack(ABC):
    """
    Base class for poisoning attacks.

    Poisoning attacks corrupt a subset of training samples before model
    training begins. Subclasses implement the concrete trigger-injection
    or label-manipulation strategy.

    Attributes:
        random_seed: int
            Seed used when randomly selecting samples to poison.
    """

    def __init__(self, random_seed: int):
        self.random_seed = random_seed

    @abstractmethod
    def poison_train(self, dataset) -> TensorDataset:
        pass

    @abstractmethod
    def poison_test(self, dataset) -> TensorDataset:
        pass
