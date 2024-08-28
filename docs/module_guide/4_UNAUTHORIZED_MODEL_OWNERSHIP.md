# Unauthorized Model Ownership
These risks are related to an adversary being able to "steal" a model, such that the stolen (surrogate) model has the same behavior and characteristics as the target model.
Amulet implements the model stealing attack from [ML-Doctor](https://github.com/liuyugeng/ML-Doctor/blob/main/doctor/modsteal.py), which is based on the following work: [Tramer et. al. *Stealing Machine Learning Models
via Prediction APIs*, USENIX Security, 2016](https://www.usenix.org/system/files/conference/usenixsecurity16/sec16_paper_tramer.pdf).
The best known class of defenses for such attacks is Watermarking or Fingerprinting.
Amulet implements the [Dataset Inference algorithm from cleverhans](https://github.com/cleverhans-lab/dataset-inference/tree/main) as a fingerprinting mechanism.
For watermarking, Amulet implements the [WatermarkNN algorithm](https://github.com/adiyoss/WatermarkNN), however, this is currently a work in progress.

## Attack
To run a model extraction attack, use `amulet.unauth_model_ownership.attacks.ModelExtraction`.
This attack trains a surrogate model using the original target model.
```python
import sys
import torch
from torch.utils.data import DataLoader, random_split
from amulet.unauth_model_ownership.attacks import ModelExtraction
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

adv_train_fraction = 0.5
random_seed = 123

# Load dataset and split train data for adversary
data = load_data(root_dir, dataset_name)

adv_train_size = int(adv_train_fraction * len(data.train_set))  # type: ignore[reportArgumentType]
target_train_size = len(data.train_set) - adv_train_size  # type: ignore[reportArgumentType]
generator = torch.Generator().manual_seed(random_seed)
target_train_set, adv_train_set = random_split(
    data.train_set, [target_train_size, adv_train_size], generator=generator
)

# Create data loaders
adv_train_loader = DataLoader(
    dataset=adv_train_set, batch_size=batch_size, shuffle=False
)
target_train_loader = DataLoader(
    dataset=target_train_set, batch_size=batch_size, shuffle=False
)
test_loader = DataLoader(
    dataset=data.test_set, batch_size=batch_size, shuffle=False
)

# Train Target Model
criterion = torch.nn.CrossEntropyLoss()

target_model = initialize_model(model, model_capacity, dataset_name, device)
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

# Train Surrogate Model
attack_model = initialize_model(
    model, model_capacity, dataset_name.device)
optimizer = torch.optim.Adam(attack_model.parameters(), lr=1e-3)
model_extraction = ModelExtraction(
    target_model,
    attack_model,
    optimizer,
    criterion,
    adv_train_loader,
    device,
    epochs,
)
attack_model = model_extraction.train_attack_model()

```

## Defense
### Fingerprinting
To fingerprint a target model, use `amulet.unauth_model_ownership.defenses.Fingerprinting`.
Note that unlike other modules, a *suspect* model is required to run fingerprinting, as it outputs whether the suspect model was stolen or not.
This could be a surrogate model using the attack above, or a separately trained model.

```python
import sys
import torch
from torch.utils.data import DataLoader
from amulet.unauth_model_ownership.defenses import Fingerprinting
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
model = 'vgg' # One of [vgg, linearnet, binarynet]
model_capacity = 'm1' # One of [m1, m2, m3, m4]
device = 'cuda:0'
epochs = 100

distance = 'l1' # One of ['l1', 'l2', 'linf', 'vanilla']
num_iter = 500
alpha_l1 = 1.0 # Step size for L1 attack
alpha_l2 = 0.01 # Step size for L2 attack
alpha_linf = 0.001 # Step size for Linf attack
gap = 0.001 # Required for L1 attacks
k = 1 # Required for L1 attacks
regressor_embed = 0 # One of [0, 1]. If true, uses extra distinct data points for training the confidence regressor.

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

target_model = initialize_model(model, model_capacity, dataset_name).to(device)
optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)
target_model = train_classifier(
    target_model, train_loader, criterion, optimizer, epochs, device
)

test_accuracy_target = get_accuracy(target_model, test_loader, device)
print(f'Test accuracy of target model: {test_accuracy_target}')

# Train Suspect Model
suspect_model = initialize_model(model, model_capacity, dataset_name).to(device)
optimizer = torch.optim.Adam(suspect_model.parameters(), lr=1e-3)
suspect_model = train_classifier(
    suspect_model, train_loader, criterion, optimizer, epochs, device
)

# Run Fingerprinting
num_classes_map = {"cifar10": 10, "fmnist": 10, "census": 2, "lfw": 2}
dataset_map = {"cifar10": "2D", "fmnist": "2D", "census": "1D", "lfw": "1D"}

fingerprinting = Fingerprinting(
    target_model,
    suspect_model,
    train_loader,
    test_loader,
    num_classes_map[dataset_name],
    device,
    distance,
    dataset_map[dataset_name],
    alpha_l1,
    alpha_l2,
    alpha_linf,
    k,
    gap,
    num_iter,
    regressor_embed,
    batch_size,
)
results = fingerprinting.dataset_inference()
```

### Watermarking
To watermark a target model, use `amulet.unauth_model_ownership.defenses.WatermarkNN`. This module is a work in progress.

## Metrics
### Model Extraction
Amulet provides a set of metrics to evaluate surrogate models. Use `amulet.unauth_model_ownership.metrics.evaluate_extraction`, which takes as input:
- The target model.
- The surrogate model.
- The test data loader.
- Device to run the predictions on.

And outputs a dictionary containing:
- `target_accuracy`: Test accuracy of the target model.
- `stolen_accuracy`: Test accuracy of the stolen model.
- `fidelity`: Agreement between target and stolen model.
- `correct_fidelity`: Accuracy conditioned on fidelity, i.e., when the models agree, how often are they also correct?

### Fingerprinting / Watermarking
Amulet currently does not provide metrics to evaluate fingerprinting or watermarking.
The modules output a boolean value for each model classifiying it as either "stolen" or "independently trained".
Thus, evaluating these modules requires a pipeline to train multiple surrogate models and independently trained models to evaluate the module.
This is a work in progress.
