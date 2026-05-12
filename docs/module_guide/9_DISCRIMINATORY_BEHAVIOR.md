# Discriminatory Behavior

Discriminatory behavior evaluations focus on assessing model bias and group fairness across sensitive attributes like gender, race, or age. Amulet implements an **Adversarial Debiasing** defense to mitigate these biases.

## Adversarial Debiasing Defense

To reduce bias in a model, use `amulet.discriminatory_behavior.defenses.AdversarialDebiasing`. This defense jointly trains the main classifier and a discriminator (adversary) that attempts to predict the sensitive attributes from the classifier's outputs. The classifier is trained to perform its task while making it impossible for the adversary to infer sensitive attributes.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from amulet.discriminatory_behavior.defenses import AdversarialDebiasing
from amulet.discriminatory_behavior.metrics import DiscriminatoryBehavior
from amulet.utils import load_data, initialize_model, train_classifier

# 1. Load data with sensitive attributes
data = load_data("./data", "celeba")

# 2. Prepare Sensitive DataLoaders (including z attributes)
train_set_z = TensorDataset(
    torch.from_numpy(data.x_train).float(),
    torch.from_numpy(data.y_train).long(),
    torch.from_numpy(data.z_train).float()
)
train_loader_z = DataLoader(train_set_z, batch_size=256, shuffle=True)

test_set_z = TensorDataset(
    torch.from_numpy(data.x_test).float(),
    torch.from_numpy(data.y_test).long(),
    torch.from_numpy(data.z_test).float()
)
test_loader_z = DataLoader(test_set_z, batch_size=256, shuffle=False)

# 3. Configure and run Adversarial Debiasing
lambdas = torch.Tensor([40, 40]) # Penalty weights for sensitive attributes

debiasing = AdversarialDebiasing(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    train_loader=train_loader_z,
    test_loader=test_loader_z,
    n_sensitive_attrs=data.z_train.shape[1],
    n_classes=data.num_classes,
    lambdas=lambdas,
    device=device,
    epochs=10
)

defended_model = debiasing.train_fair()
```

## Metrics

Fairness is evaluated using subgroup metrics like **Equalized Odds** and **Demographic Parity**. Use the `amulet.discriminatory_behavior.metrics.DiscriminatoryBehavior` class to measure these values.

```python
from amulet.discriminatory_behavior.metrics import DiscriminatoryBehavior

# Initialize metric calculator
fairness_metrics = DiscriminatoryBehavior(defended_model, test_loader_z, device)

# Get dictionary of results across all sensitive attributes
subgroup_results = fairness_metrics.evaluate_subgroup_metrics()

for attr_idx, metrics in subgroup_results.items():
    print(f"Subgroup {attr_idx} metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value}")
```
