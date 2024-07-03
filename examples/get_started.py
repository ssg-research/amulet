"""
Simple end-to-end guide to evaluate your model with Amulet, with and without a defense.
"""

import sys

sys.path.append("../../")
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from amulet.evasion.attacks import EvasionPGD
from amulet.evasion.defenses import AdversarialTrainingPGD
from amulet.utils import (
    load_data,
    initialize_model,
    train_classifier,
    get_accuracy,
)

# Setup constants
root_dir = Path("../")
random_seed = 123
device = "cuda:0"
epochs = 100

# Set random seeds for reproducibility
torch.manual_seed(random_seed)

# Load dataset and create data loaders
dataset = "cifar10"  # Possible options: ['cifar10', 'fmnist', 'lfw', 'census']

data = load_data(root_dir, dataset)
train_loader = DataLoader(dataset=data.train_set, batch_size=256, shuffle=False)
test_loader = DataLoader(dataset=data.test_set, batch_size=256, shuffle=False)

# Train Target Model
criterion = torch.nn.CrossEntropyLoss()

target_model = initialize_model(
    model_arch="vgg", model_capacity="m1", dataset=dataset
).to(device)
optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)

target_model = train_classifier(
    target_model, train_loader, criterion, optimizer, epochs, device
)

# Test Model
test_accuracy_target = get_accuracy(target_model, test_loader, device)
print("Test accuracy of target model: %s", test_accuracy_target)

# Run Evasion
evasion = EvasionPGD(target_model, test_loader, device, batch_size=256, epsilon=32)
adversarial_test_loader = evasion.run_evasion()
adv_accuracy = get_accuracy(target_model, adversarial_test_loader, device)
print("Adversarial accuracy of target model: %s", adv_accuracy)

# Defend model with Adversarial Training
adv_training = AdversarialTrainingPGD(
    target_model,
    criterion,
    optimizer,
    train_loader,
    device,
    epochs,
    epsilon=32,
)
defended_model = adv_training.train_model()
test_accuracy_defended = get_accuracy(defended_model, test_loader, device)
print("Test accuracy of defended model: %s", test_accuracy_defended)

# Run Evasion against defended model
evasion = EvasionPGD(defended_model, test_loader, device, batch_size=256, epsilon=32)
adversarial_test_loader = evasion.run_evasion()
adv_accuracy = get_accuracy(defended_model, adversarial_test_loader, device)
print("Adversarial accuracy of defended model: %s", adv_accuracy)
