# Discriminatory Behavior
This module focuses on evaluating a model for bias and fairness.
While there is currently no specific attack that increases bias in a model, Amulet does implement an [adversarial debiasing method](https://xebia.com/blog/towards-fairness-in-ml-with-adversarial-networks/) to decrease bias in a model.

This module only works with datasets that have sensitive attributes, such as Census and LFW.

## Defense
To run the adversarial debiasing module, use `amulet.discriminatory_behavior.defenses.AdversarialDebiasing`. This module uses adversarial training to ensure that the model outputs are independent of the sensitive attributes in the data.
```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from amulet.discriminatory_behavior.defenses import AdversarialDebiasing
from amulet.discriminatory_behavior.metrics import DiscriminatoryBehavior
from amulet.utils import (
    load_data,
    initialize_model,
    train_classifier,
    get_accuracy,
)

root_dir = './' # Make sure to set this
dataset_name = 'lfw' # One of [lfw, census]
batch_size = 256
model = 'vgg' # One of [vgg, linearnet, binarynet]
model_capacity = 'm1' # One of [m1, m2, m3, m4]
device = 'cuda:0'
epochs = 100

# Load dataset and create data loaders
data = load_data(root_dir, dataset_name)

train_loader = DataLoader(
    dataset=data.train_set, batch_size=batch_size, shuffle=False
)
test_loader = DataLoader(
    dataset=data.test_set, batch_size=batch_size, shuffle=False
)

sensitive_train_set = TensorDataset(
    torch.from_numpy(data.x_train).type(torch.float),
    torch.from_numpy(data.y_train).type(torch.long),
    torch.from_numpy(data.z_train).type(torch.float),
)

sensitive_test_set = TensorDataset(
    torch.from_numpy(data.x_test).type(torch.float),
    torch.from_numpy(data.y_test).type(torch.long),
    torch.from_numpy(data.z_test).type(torch.float),
)

sensitive_train_loader = DataLoader(
    dataset=sensitive_train_set, batch_size=batch_size, shuffle=False
)
sensitive_test_loader = DataLoader(
    dataset=sensitive_test_set, batch_size=batch_size, shuffle=False
)

# Train Target Model
criterion = torch.nn.CrossEntropyLoss()

target_model = initialize_model(
    model, model_capacity, dataset_name, device)
optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)
target_model = train_classifier(
    target_model, train_loader, criterion, optimizer, epochs, device
)


test_accuracy_target = get_accuracy(target_model, test_loader, device)
print(f'Test accuracy of target model: {test_accuracy_target}')

# Measure discriminatory behavior of target model
discr_behavior_target = DiscriminatoryBehavior(
    target_model, sensitive_test_loader, device
)
all_metrics = discr_behavior_target.evaluate_subgroup_metrics()

metrics_labelled = {}
metrics_labelled['white_non-white'] = all_metrics[0]
metrics_labelled['males_females'] = all_metrics[1]

for attribute, metrics in metrics_labelled.items():
    print(attribute)
    for metric, value in metrics.items():
        print(f'{metric}: {value}')

if (
    dataset_name == 'lfw'
):  # change lambdas manually to get better trade-off; hyperparameter tuning is hard in this
    lambdas = torch.Tensor([45, 17])
else:
    lambdas = torch.Tensor([40, 40])

group_fairness = AdversarialDebiasing(
    target_model,
    criterion,
    optimizer,
    sensitive_train_loader,
    sensitive_test_loader,
    lambdas,
    device,
    epochs,
)
defended_model = group_fairness.train_fair()

test_accuracy_defended = get_accuracy(defended_model, test_loader, device)
print(f'Test accuracy of defended model: {test_accuracy_defended}')

# Measure discriminatory behavior of defended model
discr_behavior_defended = DiscriminatoryBehavior(
    defended_model, sensitive_test_loader, device
)
all_metrics = discr_behavior_defended.evaluate_subgroup_metrics()

metrics_labelled = {}
metrics_labelled['white_non-white'] = all_metrics[0]
metrics_labelled['males_females'] = all_metrics[1]

for attribute, metrics in metrics_labelled.items():
    print(attribute)
    for metric, value in metrics.items():
        print(f'{metric}: {value}')

```

## Metrics
Unlike other metrics, Amulet implements a class to measure discriminatory behavior.
Please use `amulet.discriminatory_behavior.metrics.DiscriminatoryBehavior` to evaluate a model's risk for discriminatory behavior.
The example above demonstrates how to use this class.
