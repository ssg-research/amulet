"""VGG implementation"""

import torch
import torch.nn as nn


class VGG(nn.Module):
    """
    Builds a VGG network. Code taken from
    https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py

    Args:
        num_classes: int
            Number of output classes in the data.
        vgg_name: str
            String to define which version of vgg to build.
            Possible values: ['VGG11','VGG13','VGG16','VGG19'].
        batch_norm: bool
            Whether or not to use batch norm.
    """

    def __init__(
        self,
        num_classes: int = 10,
        layer_config: list[int | str] = [
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
        ],
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
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
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        layers += [nn.Flatten()]

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass on the neural network.

        Args:
            x: :class:`~torch.Tensor`
                Input data

        Returns:
            Output from the model of type :class:`~torch.Tensor`
        """
        out = self.features(x)
        out = self.classifier(out)
        return out

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
