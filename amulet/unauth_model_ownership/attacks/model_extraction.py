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
        criterion: :class:`~torch.nn.Module`
            Loss function for training model.
        train_laoder: :class:`~torch.utils.data.DataLoader`
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
        criterion: nn.Module,
        train_loader: DataLoader,
        device: str,
        epochs: int = 50,
    ):
        self.target_model = target_model
        self.attack_model = attack_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = device
        self.epochs = epochs

    def train_attack_model(self) -> nn.Module:
        """
        Trains attack model by extracting the target model.

        Returns:
            Stolen model of type :class:`torch.nn.Module'.
        """
        self.attack_model.train()

        for epoch in range(self.epochs):
            correct = 0
            total = 0
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                target_model_logit = self.target_model(x)
                target_model_posterior = F.softmax(target_model_logit, dim=1)
                self.optimizer.zero_grad()
                outputs = self.attack_model(x)
                loss = self.criterion(outputs, target_model_posterior)
                loss.backward()
                self.optimizer.step()

                _, predictions = torch.max(outputs, 1)
                total += y.size(0)
                correct += predictions.eq(y).sum().item()

            print(
                f"Train Epoch: {epoch} Loss: {loss.item():.6f} Acc: {correct/total*100:.2f}"  # type: ignore[reportPossiblyUnboundVariable]
            )

        return self.attack_model
