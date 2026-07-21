"""Dense neural network implementation"""

import torch
import torch.nn as nn

from .base import AmuletModel


class LinearNet(AmuletModel):
    """Build a dense (fully connected) network for multiclass classification.

    Args:
        num_features: Number of input features.
        num_classes: Number of output classes.
        hidden_layer_sizes: Width of each hidden layer; the ith value is the number
            of neurons in the ith hidden layer. Defaults to [128, 256, 128].
        batch_norm: Whether to add batch normalization after each hidden layer.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_layer_sizes: list[int] | None = None,
        batch_norm: bool = True,
    ):
        super().__init__()
        if hidden_layer_sizes is None:
            hidden_layer_sizes = [128, 256, 128]
        self.num_features = num_features
        self.num_classes = num_classes
        self.batch_norm = batch_norm
        self.hidden_layer_sizes = hidden_layer_sizes

        layers = []
        for i, hidden_size in enumerate(hidden_layer_sizes):
            if i == 0:
                layers += [nn.Flatten()]
                layers += [nn.Linear(num_features, hidden_size)]
                if batch_norm:
                    layers += [nn.BatchNorm1d(hidden_size)]
                layers += [nn.ReLU(inplace=True)]
            else:
                layers += [nn.Linear(hidden_layer_sizes[i - 1], hidden_size)]
                if batch_norm:
                    layers += [nn.BatchNorm1d(hidden_size)]
                layers += [nn.ReLU(inplace=True)]

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_layer_sizes[-1], num_classes)

    def forward(self, x: torch.Tensor):
        """Run the forward pass and return classification logits.

        Args:
            x: Input batch.

        Returns:
            A (batch, num_classes) logits tensor.
        """
        hidden_out = self.features(x)
        return self.classifier(hidden_out)

    def get_hidden(self, x: torch.Tensor):
        """Return the final hidden-layer activations.

        Args:
            x: Input batch.

        Returns:
            A (batch, hidden_layer_sizes[-1]) tensor of hidden-layer features.
        """
        return self.features(x)
