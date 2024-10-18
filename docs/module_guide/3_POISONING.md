# Data Poisoning
Amulet implements [BadNets](https://github.com/Kooscii/BadNets) to poison a model.
To defend against these attacks, Amulet implements outlier removal using [Shapely Values](https://github.com/AI-secure/KNN-shapley).

## Attack
To run a data poisoning attack, use `amulet.poisoning.attacks.BadNets`.
This attack returns a poisoned dataset to train a model.

```python
import sys
import torch
from torch.utils.data import DataLoader
from amulet.poisoning.attacks import BadNets
from amulet.utils import (
    load_data,
    initialize_model,
    train_classifier,
    get_accuracy,
)

if len(sys.argv) > 1:
    root_dir = sys.argv[1]
else:
    root_dir = './'
dataset_name = 'cifar10' # One of [cifar10, fmnist, census, lfw]
batch_size = 256
model = 'vgg' # One of [vgg, linearnet]
model_capacity = 'm1' # One of [m1, m2, m3, m4]
device = 'cuda:0'
epochs = 100

trigger_label = 0 # The label for the poisoned data points
poisoned_portion = 0.1 # Float between 0 and 1
random_seed = 0

# Load dataset and create data loaders
data = load_data(root_dir, dataset_name)
target_train_loader = DataLoader(
    dataset=data.train_set, batch_size=batch_size, shuffle=False
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
    target_model,
    target_train_loader,
    criterion,
    optimizer,
    epochs,
    device,
)

test_accuracy_target = get_accuracy(target_model, test_loader, device)

# Poison Model
poisoned_model = initialize_model(
    model, model_capacity, data.num_features, data.num_classes
).to(device)
poisoned_model.load_state_dict(target_model.state_dict())
optimizer = torch.optim.Adam(poisoned_model.parameters(), lr=1e-3)
poisoning = BadNets(
    trigger_label,
    poisoned_portion,
    dataset_name,
    random_seed,
)

poisoned_train_set = poisoning.poison_dataset(data.train_set)
poisoned_train_loader = DataLoader(
    dataset=poisoned_train_set, batch_size=batch_size, shuffle=False
)

poisoned_model = train_classifier(
    poisoned_model,
    poisoned_train_loader,
    criterion,
    optimizer,
    epochs,
    device,
)

poisoned_test_set = poisoning.poison_dataset(data.test_set, mode="test")
poisoned_test_loader = DataLoader(
    dataset=poisoned_test_set, batch_size=batch_size, shuffle=False
)

print(
    "Target Model on Origin Data: %s",
    get_accuracy(target_model, test_loader, device),
)
print(
    "Target Model on Poisoned Data: %s",
    get_accuracy(target_model, poisoned_test_loader, device),
)
print(
    "Poisoned Model on Origin Data: %s",
    get_accuracy(poisoned_model, test_loader, device),
)
print(
    "Poisoned Model on Poisoned Data: %s",
    get_accuracy(poisoned_model, poisoned_test_loader, device),
)
```

## Defense
To run the outlier removal algorithm, use `amulet.poisoning.defenses`.
Note that this is not a published defense, however, the KNN Shapley algorithm is published and used to calculate the value of each data point.
Please see: [Ruoxi et. al., *Efficient task-specific data valuation for nearest neighbor algorithms*, ACM VLDB, 2019](https://dl.acm.org/doi/10.14778/3342263.3342637)

```python
import sys
import torch
from torch.utils.data import DataLoader
from amulet.poisoning.defenses import OutlierRemoval
from amulet.utils import (
    load_data,
    initialize_model,
    train_classifier,
    get_accuracy,
)

if len(sys.argv) > 1:
    root_dir = sys.argv[1]
else:
    root_dir = './'
dataset_name = 'cifar10' # One of [cifar10, fmnist, census, lfw]
batch_size = 256
model = 'vgg' # One of [vgg, linearnet]
model_capacity = 'm1' # One of [m1, m2, m3, m4]
device = 'cuda:0'
epochs = 100

percent = 10 # Percentage of outliers to remove

# Load dataset and create data loaders
data = load_data(root_dir, dataset_name, )
train_loader = DataLoader(
    dataset=data.train_set, batch_size=batch_size, shuffle=False
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

# Train model with Outlier Removal
outlier_removal = OutlierRemoval(
    target_model,
    criterion,
    optimizer,
    train_loader,
    test_loader,
    device,
    epochs=epochs,
    batch_size=batch_size,
    percent=percent,
)
defended_model = outlier_removal.train_model()

test_accuracy_outlier_removed = get_accuracy(
    defended_model, test_loader, device
)
print(f'Test accuracy of model with outliers removed: {test_accuracy_outlier_removed}')
```

## Metrics
Accuracy (`amulet.utils.get_accuracy`) is currently the only metric implemeneted for data poisoning.
