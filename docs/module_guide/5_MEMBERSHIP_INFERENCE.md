# Membership Inference
Amulet implements the [Likelihood Ratio Attack (LiRA)](https://openreview.net/pdf?id=inPTplK-O6V), with the implementation taken from the [Canary in Coalmine](https://github.com/YuxinWenRick/canary-in-a-coalmine) codebase.
To defend against these attacks, Amulet implements differentially private training using DP-SGD.

## Attack
To run a membership inference attack, use `amulet.membership_inference.attacks.LiRA`.
This attack classifies each data point as `in` or `out`.

```python
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from amulet.membership_inference.attacks import LiRA
from amulet.membership_inference.metrics import get_fixed_auc
from amulet.utils import (
    load_data,
    initialize_model,
    train_classifier,
    create_dir,
    get_accuracy,
)

if len(sys.argv) > 1:
    root_dir = sys.argv[1]
else:
    root_dir = './' dataset_name = 'cifar10' # One of [cifar10, fmnist, census, lfw]
batch_size = 256
model = 'vgg' # One of [vgg, linearnet, binarynet]
model_capacity = 'm1' # One of [m1, m2, m3, m4]
device = 'cuda:0'
epochs = 100

pkeep = 0.5 # Proportion of training data to use.
random_seed = 123
num_shadow = 8 # Number of shadow models to train for membership inference
num_aug = 10 # Number of images to augment

# Load dataset and create data loaders
data = load_data(root_dir, dataset_name)

dataset_size: int = len(data.train_set)  # type: ignore[reportArgumentType]

keep = np.random.choice(
    dataset_size,
    size=int(pkeep * dataset_size),
    replace=False,
)
keep.sort()
target_train_set = Subset(
    data.train_set, list(keep)
)  # ndarray not considered a Sequence
train_loader = DataLoader(
    dataset=target_train_set, batch_size=batch_size, shuffle=False
)
test_loader = DataLoader(
    dataset=data.test_set, batch_size=batch_size, shuffle=False
)

# Train Target Model
criterion = torch.nn.CrossEntropyLoss()

target_model = initialize_model(
    model, model_capacity, data.num_features, data.num_classes
).to(device)
optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)
target_model = train_classifier(
    target_model, train_loader, criterion, optimizer, epochs, device
)

test_accuracy_target = get_accuracy(target_model, test_loader, device)
print(f'Test accuracy of target model: {test_accuracy_target}')

# Run Membership Inference attack

shadow_model_dir = root_dir / 'membership_inference' / 'shadow_models'
create_dir(shadow_model_dir)

mem_inf = LiRA(
    target_model,
    keep,
    model,
    model_capacity,
    data.train_set,
    dataset_name,
    data.num_features,
    data.num_classes,
    pkeep,
    criterion,
    num_shadow,
    num_aug,
    epochs,
    device,
    shadow_model_dir,
    random_seed,
)

results = mem_inf.run_membership_inference()

print(get_fixed_auc(results['lira_online_preds'], results['true_labels']))
print(get_fixed_auc(results['lira_offline_preds'], results['true_labels']))


```

## Defense
To train a model using DP-SGD, use `amulet.membership_inference.defenses.DPSGD`.
Note that the current implementation of DP-SGD does not work with batch normalization.

```python
import sys
import torch
from torch.utils.data import DataLoader
from amulet.membership_inference.defenses import DPSGD
from amulet.utils import (
    load_data,
    initialize_model,
    train_classifier,
    get_accuracy,
)

if len(sys.argv) > 1:
    root_dir = sys.argv[1]
else:
    root_dir = './' dataset_name = 'cifar10' # One of [cifar10, fmnist, census, lfw]
batch_size = 256
model = 'vgg' # One of [vgg, linearnet, binarynet]
model_capacity = 'm1' # One of [m1, m2, m3, m4]
device = 'cuda:0'
epochs = 100

delta = 1e-5 # Target delta
max_per_sample_grad_norm = 1.0 # Clip per sample gradients to this norm
sigma = 1.0 # Noise multiplier
secure_rng = False # Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost

# Load dataset and create data loaders
data = load_data(root_dir, dataset_name)
train_loader = DataLoader(
    dataset=data.train_set, batch_size=batch_size, shuffle=False
)
test_loader = DataLoader(
    dataset=data.test_set, batch_size=batch_size, shuffle=False
)

# Train or Load Target Model
criterion = torch.nn.CrossEntropyLoss()
target_model = initialize_model(
    model, model_capacity, data.num_features, data.num_classes
).to(device)
optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)
target_model = train_classifier(
    target_model, train_loader, criterion, optimizer, epochs, device
)

test_accuracy_target = get_accuracy(target_model, test_loader, device)
print(f'Test accuracy of target model: {test_accuracy_target}')

defended_model = initialize_model(
    model, model_capacity, dataset_name, batch_norm=False
).to(device)
optimizer = torch.optim.Adam(defended_model.parameters(), lr=1e-3)
dp_training = DPSGD(
    defended_model,
    criterion,
    optimizer,
    train_loader,
    device,
    delta,
    max_per_sample_grad_norm,
    sigma,
    secure_rng,
    epochs,
)
defended_model = dp_training.train_model()

test_accuracy_defended = get_accuracy(defended_model, test_loader, device)
print(f'Test accuracy of defended model: {test_accuracy_defended}')
```

## Metrics
Amulet implements the fixed AUC method recommended by [LiRA](https://openreview.net/pdf?id=inPTplK-O6V).
Use `amulet.membership_inference.metrics.get_fixed_auc`.
This function takes as input the in/out labels and compares them with the true labels.

To evaluate the effectiveneess of the DP-SGD algorithm, we recommend using the LiRA attack.
DP-SGD often has a large impact on the test accuracy, which can be used as an evaluation metric.
