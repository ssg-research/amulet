"""VGG implementation"""

import torch
import torch.nn as nn

from .base import AmuletModel


class VGG(AmuletModel):
    """Build a VGG-style convolutional network.

    Code adapted from
    https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py.

    Args:
        num_classes: Number of output classes in the data.
        layer_config: Feature-extractor specification. Each int is a convolution's
            output channel count and each "M" inserts a max-pool layer. Defaults to a
            built-in VGG-style configuration.
        batch_norm: Whether to add batch normalization after each convolution.
    """

    def __init__(
        self,
        num_classes: int = 10,
        layer_config: list[int | str] | None = None,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if layer_config is None:
            layer_config = [
                64,
                "M",
                128,
                "M",
                256,
                256,
                "M",
                512,
                512,
                "M",
                512,
                512,
                "M",
            ]
        self.batch_norm = batch_norm
        self.classifier = nn.Linear(512, num_classes)
        self.features = self._make_layers(layer_config)

    def _make_layers(self, cfg: list[str | int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if self.batch_norm:
                    layers += [
                        nn.Conv2d(in_channels, x, kernel_size=3, padding=1),  # type: ignore[reportArgumentType]
                        nn.BatchNorm2d(x),  # type: ignore[reportArgumentType]
                        nn.ReLU(inplace=True),
                    ]
                else:
                    layers += [
                        nn.Conv2d(in_channels, x, kernel_size=3, padding=1),  # type: ignore[reportArgumentType]
                        nn.ReLU(inplace=True),
                    ]

                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        layers += [nn.Flatten()]

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return classification logits.

        Args:
            x: Input image batch.

        Returns:
            A (batch, num_classes) logits tensor.
        """
        out = self.features(x)
        out = self.classifier(out)
        return out

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Return the flattened feature-extractor output.

        Args:
            x: Input image batch.

        Returns:
            A (batch, features) tensor of globally pooled, flattened convolutional
            features (512 with the default configuration).
        """
        return self.features(x)
