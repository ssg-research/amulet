# Attribute Inference
Amulet implements the attribute inference attack from the work [Inferring Sensitive Attributes from Model Explanations](https://github.com/vasishtduddu/AttInfExplanations) by Duddu et. al. published at ACM CIKM 2022.

## Attack
To run an attribute inference attack, use `amulet.attribute_inference.attacks.DudduCIKM2022`.
This attack predicts the sensitive attributes for a dataset, for example, it could predict whether the data point was a *male* or *female* even when the training data did not use this attribute directly.
Attribute inference only works for datasets that have sensitive attributes.
Amulet provides the LFW and Census datasets for such use cases.

```python
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from amulet.attribute_inference.attacks import DudduCIKM2022
from amulet.attribute_inference.metrics import evaluate_attribute_inference
from amulet.utils import (
    load_data,
    initialize_model,
    train_classifier,
    get_accuracy,
)
from sklearn.model_selection import train_test_split

if len(sys.argv) > 1:
    root_dir = sys.argv[1]
else:
    root_dir = './' dataset_name = 'lfw' # One of [lfw, census]
batch_size = 256
model = 'vgg' # One of [vgg, linearnet, binarynet]
model_capacity = 'm1' # One of [m1, m2, m3, m4]
device = 'cuda:0'
epochs = 100

adv_train_fraction = 0.5 # Portion of training data used by adversary

# Load dataset and split train data for adversary
data = load_data(root_dir, dataset_name)

if data.z_train is None or data.z_test is None:
    raise Exception('Dataset has no sensitive attributes')

split_data = train_test_split(
    data.x_train, data.y_train, data.z_train, test_size=adv_train_fraction
)

(
    x_train_target,
    x_train_adv,
    y_train_target,
    _,
    _,
    z_train_adv,
) = split_data

x_train_target = np.array(x_train_target)
x_train_adv = np.array(x_train_adv)
y_train_target = np.array(y_train_target)
z_train_adv = np.array(z_train_adv)

# Create data loaders
target_train_set = TensorDataset(
    torch.from_numpy(x_train_target).type(torch.float),
    torch.from_numpy(y_train_target).type(torch.long),
)

test_set = TensorDataset(
    torch.from_numpy(data.x_test).type(torch.float),
    torch.from_numpy(data.y_test).type(torch.long),
)

target_train_loader = DataLoader(
    dataset=target_train_set, batch_size=batch_size, shuffle=False
)
test_loader = DataLoader(
    dataset=test_set, batch_size=batch_size, shuffle=False
)

criterion = torch.nn.CrossEntropyLoss()

target_model = initialize_model(
    model, model_capacity, dataset_name
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
print(f'Test accuracy of target model: {test_accuracy_target}')

# Run Attribute Inference attack
attribute_inference = DudduCIKM2022(
    target_model, x_train_adv, data.x_test, z_train_adv, device
)
predictions = attribute_inference.attack_predictions()

results = evaluate_attribute_inference(data.z_test, predictions)

print(results)
```

## Defense
Amulet does not currently implement any direct defenses for attribute inference.
[DP-SGD](https://github.com/ssg-research/amulet/blob/main/docs/module_guide/5_MEMBERSHIP_INFERENCE.md#defense) may be used as a general privacy preserving mechanism.

## Metrics
Use `amulet.attribute_inference.metrics.evaluate_attribute_inference` to evaluate attribute inference attacks.
This calculates the balanced accuracy and the auc score for the predictions.
