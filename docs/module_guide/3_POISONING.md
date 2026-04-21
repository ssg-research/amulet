# Data Poisoning

Data poisoning attacks involve an adversary injecting malicious samples into a model's training set. Amulet implements the BadNets backdoor attack and provides an outlier-removal defense based on KNN Shapley values.

## Attack

To run a data poisoning attack, use `amulet.poisoning.attacks.BadNets`. This attack embeds a trigger into a portion of the training set and relabels those samples to a target class.

```python
import torch
from torch.utils.data import DataLoader
from amulet.poisoning.attacks import BadNets
from amulet.utils import load_data, initialize_model, train_classifier, get_accuracy

root_dir = './'
dataset_name = 'cifar10'
batch_size = 256
device = 'cuda:0'
epochs = 10
trigger_label = 0
poisoned_portion = 0.1
random_seed = 42

# 1. Load data
data = load_data(root_dir, dataset_name)
target_train_loader = DataLoader(data.train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(data.test_set, batch_size=batch_size, shuffle=False)

# 2. Configure Poisoning Attack
poisoning = BadNets(
    trigger_label=trigger_label,
    portion=poisoned_portion,
    random_seed=random_seed,
    dataset_type="image" # or "tabular" for census
)

# 3. Poison the Training Set
poisoned_train_set = poisoning.poison_train(data.train_set)
poisoned_train_loader = DataLoader(poisoned_train_set, batch_size=batch_size, shuffle=True)

# 4. Train Poisoned Model
poisoned_model = initialize_model("vgg", "m1", data.num_features, data.num_classes).to(device)
optimizer = torch.optim.Adam(poisoned_model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

poisoned_model = train_classifier(
    poisoned_model, poisoned_train_loader, criterion, optimizer, epochs, device
)

# 5. Evaluate Attack Success (on poisoned test set)
poisoned_test_set = poisoning.poison_test(data.test_set)
poisoned_test_loader = DataLoader(poisoned_test_set, batch_size=batch_size, shuffle=False)

clean_acc = get_accuracy(poisoned_model, test_loader, device)
attack_success = get_accuracy(poisoned_model, poisoned_test_loader, device)

print(f"Clean Test Accuracy: {clean_acc}%")
print(f"Attack Success Rate (ASR): {attack_success}%")
```

## Defense

To defend against poisoning, use `amulet.poisoning.defenses.OutlierRemoval`. This module identifies and removes outliers from the training set using KNN Shapley values before retraining the model.

```python
from amulet.poisoning.defenses import OutlierRemoval

# Initialize Outlier Removal Defense
outlier_removal = OutlierRemoval(
    model=poisoned_model,
    criterion=criterion,
    optimizer=optimizer,
    train_loader=poisoned_train_loader,
    test_loader=test_loader,
    device=device,
    percent=10 # Remove the bottom 10% of samples by Shapley score
)

# Train the Robust Model
defended_model = outlier_removal.train_robust()

# Evaluate Improved Robustness
defended_asr = get_accuracy(defended_model, poisoned_test_loader, device)
print(f"ASR after defense: {defended_asr}%")
```

## Metrics

The primary metric for poisoning attacks is the **Attack Success Rate (ASR)**, which is the model's accuracy on the poisoned test set (i.e., how often it predicts the target trigger label when the trigger is present). Standard classification accuracy on clean data is also monitored to ensure the defense does not degrade performance.
