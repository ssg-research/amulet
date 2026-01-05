"""PyTorch ResNet"""

from typing import Callable

import torch
import torchvision
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)
from torchvision.models._api import WeightsEnum


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResNet(nn.Module):
    def __init__(
        self,
        size: str = "resnet18",
        pretrained: bool = False,
        num_classes: int = 10,
        replace_first: bool = True,
    ):
        super().__init__()
        self.model_map: dict[str, Callable[..., nn.Module]] = {
            "resnet18": torchvision.models.resnet18,
            "resnet34": torchvision.models.resnet34,
            "resnet50": torchvision.models.resnet50,
            "resnet101": torchvision.models.resnet101,
            "resnet152": torchvision.models.resnet152,
        }
        self.weights_map: dict[str, type[WeightsEnum]] = {
            "resnet18": ResNet18_Weights,
            "resnet34": ResNet34_Weights,
            "resnet50": ResNet50_Weights,
            "resnet101": ResNet101_Weights,
            "resnet152": ResNet152_Weights,
        }

        if pretrained:
            weights = self.weights_map[size]
        else:
            weights = None

        self.features = self.model_map[size](weights=weights)
        if replace_first:
            self.features.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.features.maxpool = nn.Identity()  # Remove aggressive downsampling

        self.features.fc = Identity()  # type: ignore[reportAttributeAccessIssue]

        if size in ["resnet18", "resnet34"]:
            self.classifier = nn.Linear(512, num_classes)
        else:
            layers = [
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, num_classes),
            ]
            self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass on the neural network.

        Args:
            x: :class:`~torch.Tensor`
                Input data

        Returns:
            Output from the model of type :class:`~torch.Tensor`
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gets the intermediate layer output from the model.

        Args:
            x: :class:`~torch.Tensor`
                Input data to the model.

        Returns:
            Output from the model of type :class:`~torch.Tensor`.
        """
        return self.features(x)
