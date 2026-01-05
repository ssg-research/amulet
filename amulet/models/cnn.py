import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
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
        conv_channels_kernel: list[tuple[int, int]] = [(20, 5), (50, 5)],
        fc_layers: list[int] = [500],
        num_classes: int = 10,
        input_channels: int = 1,
    ):
        """
        Args:
            conv_channels_kernel: List of tuples (channels, kernel_size)
                e.g., [(20,5), (50,5)] for BadNets default conv layers.
            fc_layers: List of int
                Sizes of fully connected layers before the final output.
                e.g., [500]
            num_classes: int
                Number of classes for output layer.
            input_channels: int
                Number of input channels (1 for MNIST).
        """
        super().__init__()

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

        # Calculate flattened feature size after convs
        spatial_size = 28
        for _ in conv_channels_kernel:
            spatial_size = spatial_size // 2  # Each MaxPool halves spatial size

        feature_dim = in_channels * (spatial_size**2)

        # Fully connected layers
        fc_layers_full = [feature_dim] + fc_layers
        fc_modules = []
        for i in range(len(fc_layers)):
            fc_modules.append(nn.Linear(fc_layers_full[i], fc_layers_full[i + 1]))
            fc_modules.append(nn.ReLU(inplace=True))

        # Final classifier layer
        fc_modules.append(nn.Linear(fc_layers_full[-1], num_classes))

        self.classifier = nn.Sequential(*fc_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(x, 1)
