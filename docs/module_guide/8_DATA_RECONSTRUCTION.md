# Data Reconstruction
Amulet implements the data reconstruction attack from the [ML-Doctor library](https://github.com/liuyugeng/ML-Doctor/blob/main/doctor/modinv.py) which is based off of the work [Model Inversion Attacks that Exploit Confidence Information
and Basic Countermeasures](https://rist.tech.cornell.edu/papers/mi-ccs.pdf) by Fredrikson et. al. published at ACM CCS 2015.

## Attack
To run an data reconstruction attack, use `amulet.data_reconstruction.attacks.FredriksonCCS2015`.

```python
import torch
from torch.utils.data import DataLoader
from amulet.data_reconstruction.attacks import FredriksonCCS2015
from amulet.data_reconstruction.metrics import reverse_mse
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
alpha = 3000 # Number of iterations for data reconstruction

# Load dataset and create data loaders
data = load_data(root_dir, dataset_name)
train_loader = DataLoader(dataset=data.train_set, batch_size=1, shuffle=False)
test_loader = DataLoader(dataset=data.test_set, batch_size=1, shuffle=False)

# Train or Load Target Model
criterion = torch.nn.CrossEntropyLoss()

target_model = initialize_model(
    model, model_capacity, dataset_name
).to(device)
optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)
target_model = train_classifier(
    target_model, train_loader, criterion, optimizer, epochs, device
)

test_accuracy_target = get_accuracy(target_model, test_loader, device)
print(f'Test accuracy of target model: {test_accuracy_target}')

# Run Data Reconstruction attack
input_size = (1,) + tuple(data.test_set[0][0].shape)
num_classes_dict = {'cifar10': 10, 'fmnist': 10, 'census': 2, 'lfw': 2}
output_size = num_classes_dict[dataset_name]
data_recon = FredriksonCCS2015(
    target_model, input_size, output_size, device, alpha
)

reverse_data = data_recon.get_reconstructed_data()

mse_loss = reverse_mse(
    test_loader, reverse_data, input_size, output_size, device
)

print(f'Reverse MSE: {mse_loss}')
```

## Defense
Amulet does not currently implement any direct defenses for data reconstruction.
[DP-SGD](https://github.com/ssg-research/amulet/blob/main/docs/module_guide/5_MEMBERSHIP_INFERENCE.md#defense) may be used as a general privacy preserving mechanism.

## Metrics
To evaluate data reconstruction attacks, use `amulet.data_reconstruction.metrics.reverse_mse`.
This function calculates the *average* data point in each class, and finds the class-wise MSE between the reconstructed (reverse) data point and the average.
This class-wise MSE is then averaged over all classes.
