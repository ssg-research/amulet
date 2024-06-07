"""Simple CNN classifier implementation
"""
from typing import List

import torch
from torch import nn

class CNN(nn.Module): 
    """
        Builds a neural network for a simply cnn classification task.

    """
    def __init__(self):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),
            nn.Tanh(),
            nn.Conv2d(6, 16, 5),
            nn.Tanh(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 10 * 10, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
        )  

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 10 * 10)
        return self.classifier(x)
    