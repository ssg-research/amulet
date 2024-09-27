"""Dense neural network implementation"""

import torch
import torch.nn as nn


class LinearNet(nn.Module):
    """
    Builds a dense neural network for a multiclass image classification task.

    Args:
        num_features: int
            Number of features of input data.
        hidden_layer_size: List of int
            The ith number represents the number of neurons
            in the ith hidden layer.
        num_classes: int
            Number of output classes.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_layer_sizes: list[int] = [128, 256, 128],
    ):
        super().__init__()

        layers = []
        for i, hidden_size in enumerate(hidden_layer_sizes):
            if i == 0:
                layers += [nn.Flatten()]
                layers += [nn.Linear(num_features, hidden_size)]
                layers += [nn.Tanh()]
            else:
                layers += [nn.Linear(hidden_layer_sizes[i - 1], hidden_size)]
                layers += [nn.Tanh()]

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_layer_sizes[-1], num_classes)

    def forward(self, x: torch.Tensor):
        """
        Runs the forward pass on the neural network.

        Args:
            x: :class:`~torch.Tensor`
                Input data

        Returns:
            Output from the model of type :class:`~torch.Tensor`
        """
        hidden_out = self.features(x)
        return self.classifier(hidden_out)

    def get_hidden(self, x: torch.Tensor):
        """
        Gets the intermediate layer output from the model.

        Args:
            x: :class:`~torch.Tensor`
                Input data to the model.

        Returns:
            Output from the model of type :class:`~torch.Tensor`.
        """
        return self.features(x)
