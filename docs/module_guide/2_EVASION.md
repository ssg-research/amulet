# Evasion

Evasion attacks (adversarial examples) occur when an adversary adds carefully crafted perturbations to inputs to cause a model to misclassify them. Amulet implements the **Projected Gradient Descent (PGD)** algorithm for both attacks and adversarial training defenses.

## Attack

To run an evasion attack, use `amulet.evasion.attacks.EvasionPGD`. This attack produces a `DataLoader` containing adversarial examples generated from a given test set.

```python
from amulet.evasion.attacks import EvasionPGD
from amulet.utils import load_data, initialize_model, train_classifier, get_accuracy

# 1. Load data and train target model
data = load_data("./data", "cifar10")
target_model = initialize_model("vgg", "m1", data.num_features, data.num_classes).to(device)
target_model = train_classifier(target_model, train_loader, criterion, optimizer, 10, device)

# 2. Configure and run Evasion attack
evasion = EvasionPGD(
    model=target_model,
    test_loader=test_loader,
    device=device,
    batch_size=256,
    epsilon=0.01 # Perturbation budget
)

adv_loader = evasion.attack()

# 3. Evaluate results
clean_acc = get_accuracy(target_model, test_loader, device)
adv_acc = get_accuracy(target_model, adv_loader, device)

print(f"Clean Accuracy: {clean_acc}%")
print(f"Adversarial Accuracy: {adv_acc}%")
```

## Defense

To defend against evasion attacks, use `amulet.evasion.defenses.AdversarialTrainingPGD`. This defense trains the model on a combination of clean and adversarial samples generated during training.

```python
from amulet.evasion.defenses import AdversarialTrainingPGD

# Configure and run Adversarial Training
adv_training = AdversarialTrainingPGD(
    model=target_model,
    criterion=criterion,
    optimizer=optimizer,
    train_loader=train_loader,
    device=device,
    epochs=10,
    epsilon=0.01
)

defended_model = adv_training.train_robust()
```

## Metrics

The primary metric for evasion attacks is **Adversarial Accuracy**, which measures the model's performance on inputs perturbed by the attacker. Robustness is evaluated by comparing the adversarial accuracy of an undefended model versus a defended one. Use `amulet.utils.get_accuracy` to calculate these values.
