# Extending Amulet with a Custom Risk

This example demonstrates how to add a new **risk** to Amulet.  
A risk encapsulates one or more attacks and defines how a model can be evaluated under a new threat model.

## Step 1: Create a New Risk Directory

Create a new directory under `amulet/` for the risk:

```
amulet/test_time_adaptation/
amulet/test_time_adaptation/attacks/
```

## Step 2: Implement the Attack

Add a new attack file under the `attacks` subdirectory.

**File:** `amulet/test_time_adaptation/attacks/test_time_data_poisoning.py`

```python
import torch
import torch.nn as nn

class TestTimeDataPoisoning:
    def __init__(self, target_model: nn.Module, test_loader, device):
        self.model = target_model
        self.test_loader = test_loader
        self.device = device

    def attack(self) -> torch.utils.data.TensorDataset:
        # Adapt model parameters using adversarial test-time inputs
        ...
```

## Step 3: Expose the Attack

Add an `__init__.py` file to expose the attack class.

**File:** `amulet/test_time_adaptation/attacks/__init__.py`

```python
from .test_time_data_poisoning import TestTimeDataPoisoning

__all__ = ["TestTimeDataPoisoning"]
```

At this point, the new risk and attack are fully defined and can be imported by downstream code.
