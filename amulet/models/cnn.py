import torch
import torch.nn as nn

from .base import AmuletModel


class SimpleCNN(AmuletModel):
    """
    Parameterized CNN for MNIST, similar to the one used in the BadNets paper.

    Reference:
        BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain
        Tianyu Gu, Brendan Dolan-Gavitt, Siddharth Garg
        https://arxiv.org/abs/1708.06733.

    Architecture:
    - Multiple Conv2d layers (with ReLU + MaxPool)
    - Multiple Fully Connected layers
    """

    def __init__(
        self,
        conv_channels_kernel: list[tuple[int, int]] | None = None,
        fc_layers: list[int] | None = None,
        num_classes: int = 10,
        input_channels: int = 1,
    ):
        """Build the CNN's convolutional stack and fully connected classifier.

        Args:
            conv_channels_kernel: (channels, kernel_size) per convolution layer,
                e.g. [(20, 5), (50, 5)] for the BadNets default. Defaults to that.
            fc_layers: Widths of the fully connected layers before the final output,
                e.g. [500]. Defaults to [500].
            num_classes: Number of classes for the output layer.
            input_channels: Number of input channels (1 for MNIST).
        """
        super().__init__()

        if conv_channels_kernel is None:
            conv_channels_kernel = [(20, 5), (50, 5)]
        if fc_layers is None:
            fc_layers = [500]

        layers = []
        in_channels = input_channels

        # Conv layers with ReLU + MaxPool2d(2)
        for out_channels, kernel_size in conv_channels_kernel:
            padding = kernel_size // 2  # same padding
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
            )
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        # Calculate flattened feature size after convs.
        # Hardcoded for 28x28 inputs (MNIST/FMNIST); does not generalize to other resolutions.
        spatial_size = 28
        for _ in conv_channels_kernel:
            spatial_size = spatial_size // 2  # Each MaxPool halves spatial size

        feature_dim = in_channels * (spatial_size**2)

        # Fully connected layers
        fc_layers_full = [feature_dim, *fc_layers]
        fc_modules = []
        for i in range(len(fc_layers)):
            fc_modules.append(nn.Linear(fc_layers_full[i], fc_layers_full[i + 1]))
            fc_modules.append(nn.ReLU(inplace=True))

        # Final classifier layer
        fc_modules.append(nn.Linear(fc_layers_full[-1], num_classes))

        self.classifier = nn.Sequential(*fc_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return classification logits.

        Args:
            x: Input image batch.

        Returns:
            A (batch, num_classes) logits tensor.
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Return the flattened convolutional feature map.

        Args:
            x: Input image batch.

        Returns:
            A (batch, features) tensor: the conv stack output flattened per sample.
        """
        x = self.features(x)
        return torch.flatten(x, 1)
