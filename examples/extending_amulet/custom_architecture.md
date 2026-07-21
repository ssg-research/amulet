# Extending Amulet with a Custom Model Architecture

This example demonstrates how to add a new **model architecture** to Amulet.
All models follow a simple, standardized interface: every model subclasses
`AmuletModel` (`amulet/models/base.py`) and implements both `forward` and
`get_hidden`. Several modules (e.g. `get_intermediate_features`, `OutlierRemoval`)
rely on `get_hidden` returning intermediate activations; subclassing `AmuletModel`
enforces that contract at runtime instead of failing silently.

## Step 1: Implement the Model

Create a new model file. Subclass `AmuletModel`, not `nn.Module` directly.

**File:** `amulet/models/vit.py`

```python
import torch

from amulet.models.base import AmuletModel

class VisionTransformer(AmuletModel):
    def __init__(self, **hyperparameters):
        super().__init__()
        # Initialize Vision Transformer layers
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        # Return the pooled intermediate representation (e.g. the CLS/last hidden
        # state). Required: omitting it raises NotImplementedError at runtime.
        ...
```

For a full worked example that subclasses `AmuletModel` around a real pretrained
backbone, see `amulet/models/hf_causal_lm.py` (`HFCausalLM`), which wraps a Hugging Face
causal (decoder-only) LM with LoRA and exposes classification, perplexity scoring, and
generation over one shared adapted decoder.

## Step 2: Export the Model

Update the models namespace.

**File:** `amulet/models/__init__.py`

```python
from .base import AmuletModel
from .cnn import SimpleCNN
from .hf_causal_lm import HFCausalLM
from .linear_net import LinearNet
from .resnet import ResNet
from .vgg import VGG
from .vit import VisionTransformer

__all__ = [
    "VGG",
    "AmuletModel",
    "HFCausalLM",
    "LinearNet",
    "ResNet",
    "SimpleCNN",
    "VisionTransformer",
]
```

## Step 3: Register the Model in the Pipeline

Add a new case in the model initialization pipeline (`initialize_model` in `amulet/utils/__pipeline.py`).

```python
elif model_arch == "vit":
    model = VisionTransformer(
        num_classes=num_classes
    )
```

No changes to attacks, defenses, or metrics are required when adding a new model.
