# Amulet
Amulet is a Python machine learning (ML) package to evaluate the susceptibility of different risks to security, privacy, and fairness. Amulet is applicable to evaluate how algorithms designed to reduce one risk may impact another unrelated risk and compare different attacks/defenses for a given risk.

Amulet builds upon prior work titled [“SoK: Unintended Interactions among Machine Learning Defenses and Risks”](https://arxiv.org/abs/2312.04542) which appears in IEEE Symposium on Security and Privacy 2024. The SoK covers only two interactions and identifies the design of a software library to evaluate unintended interactions as future work. Amulet addresses this gap by including eight different risks each covering their own attacks, defenses and metrics.

Amulet is:
- Comprehensive: Covers the most representative attacks/defenses/metrics for different risks.
- Extensible: Easy to include additional risks, attacks, defenses, or metrics.
- Consistent: Allows using different attacks/defenses/metrics with a consistent, easy-to-use API.
- Applicable: Allows evaluating unintended interactions among defenses and attacks.


Built to work with PyTorch, you can incorporate Amulet into your current ML pipeline to test how your model interacts with these state-of-the-art defenses and risks. Alternatively, you can use the example pipelines to bootstrap your pipeline.


## Requirements

**Note:** The package requires the CUDA version to be 11.8 or above for PyTorch 2.2

### Install poetry

`python3 -m venv .poetry_venv`

`. .poetry_venv/bin/activate` or `. .venv/bin/activate.fish`

`python -m pip install --upgrade pip`

`pip install poetry`

`deactivate`

Consider setting `.poetry_venv/bin/poetry config virtualenvs.create false` to prevent poetry from creating its own venv.

To create the virtual environment:
`python3 -m venv .venv`

To activate it:
`source .venv/bin/activate` or if using fish `. .venv/bin/activate.fish`

Then, to install the dependencies:
`.poetry_venv/bin/poetry install`

**DISCLAIMER:** Installing `pytorch` with `poetry` is [still weird](https://github.com/python-poetry/poetry/blob/main/docs/repositories.md#explicit-package-sources) but should work.

## Usage
The following minimal example runs an evasion attack:
```python
import torch
from pathlib import Path
from amulet.datasets import load_cifar10
from amulet.utils import initialize_model, train_classifier, get_accuracy
from amulet.evasion.attacks import EvasionPGD

device = 'cuda:{0}'
batch_size = 32
root_dir = Path('./')

# Load data
data_path = root_dir / 'data' / 'cifar10' 
data = load_cifar10(data_path)
train_loader = torch.utils.data.DataLoader(dataset=data.train_set, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=data.test_set, batch_size=32, shuffle=False)

# Setup model architecture
model_architecture = 'vgg'
model_capacity = 'm1' # ['m1', 'm2', m3', m4']
target_model = initialize_model('vgg', 'm1', 'cifar10').to(device)
optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# Train model
epochs = 2
target_model = train_classifier(target_model, train_loader, criterion, optimizer, epochs, device)

# Run Evasion 
evasion = EvasionPGD(target_model,
                        test_loader,
                        device,
                        batch_size,
                        epsilon = 32)
adversarial_test_loader = evasion.run_evasion()

# Evaluate metrics
test_accuracy_target = get_accuracy(target_model, test_loader, device)
print(f'Test accuracy of target model: {test_accuracy_target}')     

adv_accuracy = get_accuracy(target_model, adversarial_test_loader, device)
print(f'Robust accuracy of target model: {adv_accuracy}')     
```
For more complete examples of an end-to-end pipeline, please see our [examples](https://github.com/ssg-research/amulet/tree/main/examples)
## Features

We provide a high-level list of features below. Please refer to the Tutorial for more information.

### Datasets
Amulet provides the following for computer vision tasks:
- [CIFAR10](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html).
- [FashionMNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html).

Amulet pre-processes and provides the following datasets to test for privacy-related concerns:
- [Census Income Dataset](https://archive.ics.uci.edu/dataset/20/census+income).
- [Labeled Faces in the Wild (LFW)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html).

### Models
Amulet provides the following models:
- [VGG](https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/): for computer vision tasks.
- LinearNet: A dense neural network tuned for multiclass classification on the FashionMNIST dataset.
- BinaryNet: A dense neural network tuned for binary classification.

All models have configurable sizes to evaluate the impact of model capacity. Please see [Defining a Model]() for more details.

### Risks
Amulet provides attacks, defenses, and evaluation metrics for the following risks:
#### Security
- Evasion
- Poisoning
- Unauthorized Model Ownership

#### Privacy
- Membership Inference
- Attribute Inference
- Distribution Inference
- Data Reconstruction

#### Fairness
- Discriminatory Behavior

Please check the Module Heirarchy for more details.

