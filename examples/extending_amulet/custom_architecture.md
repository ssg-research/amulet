# Extending Amulet with a Custom Model Architecture

This example demonstrates how to add a new **model architecture** to Amulet.
All models follow a simple, standardized interface.

## Step 1: Implement the Model

Create a new model file.

**File:** `amulet/models/vit.py`

```python
import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, **hyperparameters):
        super().__init__()
        # Initialize Vision Transformer layers
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
    
    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

## Step 2: Export the Model

Update the models namespace.

**File:** `amulet/models/__init__.py`

```python
from .vgg import VGG
from .linear_net import LinearNet
from .resnet import ResNet
from .cnn import SimpleCNN
from .vit import VisionTransformer

__all__ = ["VGG", "LinearNet", "ResNet", "SimpleCNN", "VisionTransformer"]
```

## Step 3: Register the Model in the Pipeline

Add a new case in the model initialization pipeline.

```python
elif model_arch == "vit":
    model = VisionTransformer(
        num_classes=num_classes
    )
```

No changes to attacks, defenses, or metrics are required when adding a new model.
