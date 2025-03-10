"""PyTorch ResNet"""

import torch
import torchvision
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = torchvision.models.resnet18(pretrained=True)
        self.features.fc = Identity()  # type: ignore[reportAttributeAccessIssue]
        self.classifier = nn.Linear(512, num_classes)

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
        x = torch.sigmoid(x)
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
