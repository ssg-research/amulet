"""VGG implementation
"""
from typing import List, Union, Dict

import torch
import torch.nn as nn

cfgs: Dict[str, List[Union[str, int]]] = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    """
        Builds a VGG network. Code taken from 
        https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py

        Args:
            vgg_name: str
                String to define which version of vgg to build.
                Possible values: ['VGG11','VGG13','VGG16','VGG19'].
            num_classes: int
                Number of classes for the labels.
    """
    def __init__(
        self, vgg_name: str, num_classes: int = 10
    ) -> None:
        super().__init__()
        self.features = self._make_layers(cfgs[vgg_name])
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
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg: List[Union[str, int]]) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

        return nn.Sequential(*layers)

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
