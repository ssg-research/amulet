# Membership Inference

Membership inference attacks aim to determine if a specific data point was used to train a model. Amulet implements the **Likelihood Ratio Attack (LiRA)** and provides **DP-SGD** as a robust defense against such privacy risks.

## Membership Inference Attack

To run a membership inference attack, use `amulet.membership_inference.attacks.LiRA`. This attack uses shadow models trained on subsets of the original dataset to estimate the likelihood that a point was in the target model's training set.

```python
import numpy as np
from torch.utils.data import DataLoader, Subset
from amulet.membership_inference.attacks import LiRA
from amulet.membership_inference.metrics import compute_mi_metrics
from amulet.utils import load_data, initialize_model, train_classifier

# 1. Load data and identify training members
data = load_data("./data", "cifar10")
dataset_size = len(data.train_set)
in_data_indices = np.random.choice(dataset_size, size=int(0.5 * dataset_size), replace=False)

train_loader = DataLoader(Subset(data.train_set, in_data_indices), batch_size=256)

# 2. Train Target Model
target_model = initialize_model("vgg", "m1", data.num_features, data.num_classes).to(device)
optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)
target_model = train_classifier(target_model, train_loader, criterion, optimizer, 10, device)

# 3. Configure and run LiRA attack
mem_inf = LiRA(
    target_model=target_model,
    in_data=in_data_indices,
    shadow_architecture="vgg",
    shadow_capacity="m1",
    train_set=data.train_set,
    dataset="cifar10",
    num_features=data.num_features,
    num_classes=data.num_classes,
    batch_size=256,
    pkeep=0.5,
    criterion=criterion,
    num_shadow=10, # Number of shadow models
    epochs=10,
    device=device,
    models_dir="./shadow_models",
    exp_id=42
)

results = mem_inf.attack()

# 4. Evaluate Metrics
metrics = compute_mi_metrics(results['lira_online_preds'], results['true_labels'])
print(f"Attack AUC: {metrics['auc_score']}")
```

## DP-SGD Defense

To train a model with differential privacy, use `amulet.membership_inference.defenses.DPSGD`. Note that the current implementation of DP-SGD is incompatible with standard batch normalization.

```python
from amulet.membership_inference.defenses import DPSGD

# Initialize model without batch normalization
defended_model = initialize_model("vgg", "m1", data.num_features, data.num_classes, batch_norm=False).to(device)
optimizer = torch.optim.Adam(defended_model.parameters(), lr=1e-3)

# Configure and run DP-SGD
dp_training = DPSGD(
    model=defended_model,
    criterion=criterion,
    optimizer=optimizer,
    train_loader=train_loader,
    device=device,
    delta=1e-5,
    max_per_sample_grad_norm=1.0,
    sigma=1.0, # Noise multiplier
    epochs=10
)

defended_model = dp_training.train_private()
```

## Metrics

Membership inference effectiveness is evaluated using the **AUC score** and **Precision at low False Positive Rates (FPR)**, which are standard for measuring how well an adversary can distinguish members from non-members. Use `amulet.membership_inference.metrics.compute_mi_metrics` to compute these values from attack results.
