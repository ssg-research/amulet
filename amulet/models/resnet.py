"""PyTorch ResNet"""

from collections.abc import Callable

import torch
import torch.nn as nn
import torchvision
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)
from torchvision.models._api import WeightsEnum

from .base import AmuletModel


class Identity(nn.Module):
    """Pass-through module used to replace the backbone's final fully connected layer."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ResNet(AmuletModel):
    """Build a torchvision ResNet backbone with a fresh classification head.

    The backbone's final fully connected layer is replaced with an identity, so
    `features` yields pooled backbone embeddings and a separate `classifier`
    produces the class logits.

    Args:
        size: Which ResNet variant to build. One of "resnet18", "resnet34",
            "resnet50", "resnet101", "resnet152".
        pretrained: Whether to initialize the backbone with torchvision's ImageNet
            pretrained weights.
        num_classes: Number of output classes.
        replace_first: Whether to swap the initial 7x7 stride-2 convolution for a 3x3
            stride-1 convolution and drop the first max-pool, adapting the stem for
            low-resolution inputs such as CIFAR.
    """

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

        weights = self.weights_map[size] if pretrained else None

        self.features = self.model_map[size](weights=weights)
        if replace_first:
            self.features.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.features.maxpool = nn.Identity()  # Remove aggressive downsampling

        self.features.fc = Identity()

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
        """Run the forward pass and return classification logits.

        Args:
            x: Input image batch.

        Returns:
            A (batch, num_classes) logits tensor.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled backbone embedding.

        Args:
            x: Input image batch.

        Returns:
            The backbone feature embedding for each sample.
        """
        return self.features(x)
