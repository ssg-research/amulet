# Introduction

Each risk in Amulet represents either an *attack*, *defense*, or *metric*. To see a list of the risks and the features, please see the [Getting Started guide](https://github.com/ssg-research/amulet/blob/main/docs/GETTING_STARTED.md).

## Design Overview
Most attacks and defenses are designed such that they take the target model, and some additional information (such as data, hyperparameters, configuration, etc.) as input, run an algorithm, and output a result. This result can then be passed onto the respective metrics modules to evaluate the attack or defense. A brief pipeline would look something like:
```
data = load_data()

target_model = initialize_model(*model_architecture_parameters)
target_model = train_model(target_model, data.train, data.test, *training_parameters)

attack = AttackClass()
attack_output = attack.run_attack(target_model, data.test, *attack_parameters)

result = evalute_attack(attack_output)
```

The rest of this document provides detailed usage guides for each risk implemented in Amulet. Please see the [example scripts](https://github.com/ssg-research/amulet/tree/main/examples) provided for each attack / defense. These scripts provide an end-to-end pipeline that may be used to run experiments.

# Security Risks
Amulet implements evasion, data poisoning, and unauthorized model ownership.

## Evasion
Amulet implements the Projected Gradient Descent (PGD) algorithm from [cleverhans](https://github.com/cleverhans-lab/cleverhans) to generate adversarial examples. PGD is used for the evasion attack and the adversarial training defense.

### Attack:
To run an evasion attack, use amulet.evasion.attacks.EvasionPGD. This module returns a data loader containing the adversarial examples.

The following code snippet shows a brief example:
```python
import torch
from torch.utils.data import DataLoader
from amulet.evasion.attacks import EvasionPGD
from amulet.utils import (
    load_data,
    initialize_model,
    train_classifier,
    get_accuracy,
)

dataset_name = 'cifar10' # One of [cifar10, fmnist, census, lfw]
batch_size = 256
model = 'vgg' # One of [vgg, linearnet, binarynet]
model_capacity = 'm1' # One of [m1, m2, m3, m4]
device = 'cuda:0'
epochs = 100
epsilon = 32 # Controls the amount of perturbation. Divided by 255.

# Load dataset and create data loaders
data = load_data(root_dir, dataset_name)
train_loader = DataLoader(
    dataset=data.train_set, batch_size=batch_size, shuffle=False
)
test_loader = DataLoader(
    dataset=data.test_set, batch_size=batch_size, shuffle=False
)

# Train Target Model
criterion = torch.nn.CrossEntropyLoss()

target_model = initialize_model(
    model, model_capacity, dataset_name
).to(device)
optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)
target_model = train_classifier(
    target_model, train_loader, criterion, optimizer, epochs, device
)

# Run Evasion
evasion = EvasionPGD(
    target_model, test_loader, device, batch_size, epsilon
)
adversarial_test_loader = evasion.run_evasion()

# Test model
test_accuracy_target = get_accuracy(target_model, test_loader, device)
adversarial_accuracy = get_accuracy(target_model, adversarial_test_loader, device)
```

### Defense:
To run the adversarial training algorithm use amulet.evasion.defenses.AdversarialTrainingPGD. This module trains a model using adversarial training.

The following code snippet shows a brief example:
```python
import torch
from torch.utils.data import DataLoader
from amulet.evasion.defenses import AdversarialTrainingPGD
from amulet.utils import (
    load_data,
    initialize_model,
    train_classifier,
    get_accuracy,
)

dataset_name = 'cifar10' # One of [cifar10, fmnist, census, lfw]
batch_size = 256
model = 'vgg' # One of [vgg, linearnet, binarynet]
model_capacity = 'm1' # One of [m1, m2, m3, m4]
device = 'cuda:0'
epochs = 100
epsilon = 32 # Controls the amount of perturbation. Divided by 255.

# Load dataset and create data loaders
data = load_data(root_dir, dataset_name)
train_loader = DataLoader(
    dataset=data.train_set, batch_size=batch_size, shuffle=False
)
test_loader = DataLoader(
    dataset=data.test_set, batch_size=batch_size, shuffle=False
)

# Initialize Target Model
criterion = torch.nn.CrossEntropyLoss()

target_model = initialize_model(
    model, model_capacity, dataset_name
).to(device)
optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)
target_model = train_classifier(
    target_model, train_loader, criterion, optimizer, epochs, device
)

# Fine-tune target model with adversarial training
adv_training = AdversarialTrainingPGD(
    target_model,
    criterion,
    optimizer,
    train_loader,
    device,
    epochs,
    epsilon,
)
defended_model = adv_training.train_model()

test_accuracy_defended = get_accuracy(defended_model, test_loader, device)
```
