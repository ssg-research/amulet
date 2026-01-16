# Extending Amulet with a Custom Metric

This example shows how to define a **custom evaluation metric** for a new or existing risk.

Metrics in Amulet are implemented as standalone functions and grouped by risk.

## Step 1: Create a Metrics Directory

Create a new metrics directory under the corresponding risk:

```
amulet/test_time_adaptation/metrics/
```

## Step 2: Implement the Metric

For test-time adaptation, a natural metric is attack accuracy.

**File:** `amulet/test_time_adaptation/metrics/attack_accuracy.py`

```python
import torch

def attack_accuracy(model, test_loader, device) -> float:
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total
```

## Step 3: Expose the Metric

Add an `__init__.py` file.

**File:** `amulet/test_time_adaptation/metrics/__init__.py`

```python
from .attack_accuracy import attack_accuracy

__all__ = ["attack_accuracy"]
```

The metric can now be used consistently with other Amulet evaluation utilities.
