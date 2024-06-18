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

    @staticmethod
    def evaluate_attack(
        target_model: nn.Module,
        attack_model: nn.Module,
        data_loader: DataLoader,
        device: str,
    ) -> dict:
        """
        Compares the attack model with the target model with respect to the accuracy,
        fidelity (agreement) and correct fidelity (agreement on correct predictions).

        Args:
            test_loader: :class:'~torch.utils.data.DataLoader
                Input data to the models.

        Returns:
            Dictionary containing the resulting values:
                'target_accuracy': Accuracy of the target model.
                'stolen_accuracy': Accuracy of the stolen (attack) model
                'fidelity': Fidelity between target and stolen model
                'correct_fidelity': Fidelity between target and stolen model
                                    on correct predictions
        """
        target_model.eval()
        attack_model.eval()

        fidelity = 0
        target_correct = 0
        stolen_correct = 0
        both_correct = 0
        total = 0
        with torch.no_grad():
            for x, labels in data_loader:
                x, labels = x.to(device), labels.to(device)
                target_model_outputs = target_model(x)
                stolen_model_outputs = attack_model(x)
                _, target_model_predictions = torch.max(target_model_outputs.data, 1)
                _, stolen_model_predictions = torch.max(stolen_model_outputs.data, 1)

                total += target_model_predictions.size(0)
                fidelity += (
                    (target_model_predictions == stolen_model_predictions).sum().item()
                )
                target_correct += (target_model_predictions == labels).sum().item()
                stolen_correct += (stolen_model_predictions == labels).sum().item()
                both_correct += (
                    (
                        (stolen_model_predictions == labels)
                        & (target_model_predictions == labels)
                    )
                    .sum()
                    .item()
                )

        return {
            "target_accuracy": (target_correct / total) * 100,
            "stolen_accuracy": (stolen_correct / total) * 100,
            "fidelity": (fidelity / total) * 100,
            "correct_fidelity": (both_correct / total) * 100,
        }
