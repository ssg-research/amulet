# Introduction

Amulet organizes machine learning risks into three primary components: **Attacks**, **Defenses**, and **Metrics**.
The core design philosophy is to allow these components to be easily composed into a unified training and evaluation pipeline.

## Component Categories

1. **Attacks**: Algorithms that generate adversarial artifacts (like perturbed inputs or poisoned datasets) or trained surrogate models.
2. **Defenses**: Robust training algorithms or pre-processing steps designed to mitigate specific risks.
3. **Metrics**: Evaluation functions that measure the success of an attack or the robustness of a model.

## Design Overview

Most Amulet components follow a consistent API:

- **Attacks** generally take a `target_model` and a `DataLoader` as input and provide an `attack()` method.
- **Defenses** provide methods like `train_robust()`, `train_private()`, or `train_fair()` depending on the risk they address; inference-time defenses (e.g. ONION) also expose `purify(dataset)` alongside `train_robust()`.
- **Metrics** take model predictions or outputs and return standard performance scores.

### Example Pipeline

A typical Amulet pipeline involves loading data, initializing a model, applying a defense, and then running an attack to evaluate the defense's effectiveness:

```python
from amulet.utils import load_data, initialize_model, train_classifier, get_accuracy
from amulet.evasion.attacks import EvasionPGD
from amulet.evasion.defenses import AdversarialTrainingPGD

# 1. Load Data
data = load_data("./data", "cifar10")
train_loader = DataLoader(data.train_set, batch_size=128)
test_loader = DataLoader(data.test_set, batch_size=128)

# 2. Initialize and Train Target Model
model = initialize_model("vgg", "m1", data.num_features, data.num_classes)
model = train_classifier(model, train_loader, criterion, optimizer, epochs, device)

# 3. Apply Defense
defense = AdversarialTrainingPGD(model, criterion, optimizer, train_loader, device, epochs)
defended_model = defense.train_robust()

# 4. Evaluate with Attack
attack = EvasionPGD(defended_model, test_loader, device, batch_size=128)
adv_loader = attack.attack()
adv_accuracy = get_accuracy(defended_model, adv_loader, device)

print(f"Adversarial Accuracy: {adv_accuracy}%")
```

For more details, check the specific guide for each risk and the provided [example scripts](https://github.com/ssg-research/amulet/tree/main/examples).
