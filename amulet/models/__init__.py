"""
The module amulet.models includes utilities to build sample models.
"""

from .base import AmuletModel
from .cnn import SimpleCNN
from .linear_net import LinearNet
from .resnet import ResNet
from .vgg import VGG

__all__ = ["VGG", "AmuletModel", "LinearNet", "ResNet", "SimpleCNN"]
