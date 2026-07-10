# Getting Started

Amulet (`amuletml` on PyPI) is a PyTorch-based research library for evaluating unintended interactions among machine learning (ML) defenses and risks across security, privacy, and fairness.

## Features

### Datasets

Amulet provides built-in support for several common datasets, including automated downloading and pre-processing:

- **Computer Vision**: [CIFAR-10](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html), [CIFAR-100](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR100.html), [FashionMNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html), [MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html).
- **Face Attributes**: [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [Labeled Faces in the Wild (LFW)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html), [UTKFace](https://susanqq.github.io/UTKFace/).
- **Tabular Data**: [Census Income Dataset](https://archive.ics.uci.edu/dataset/20/census+income).
- **Text**: [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2), [AG News](https://huggingface.co/datasets/fancyzhx/ag_news), [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) (loaded via Hugging Face `datasets`; requires the optional `llm` extra).

### Models

Amulet provides pre-configured architectures with scalable capacity:

- **VGG**: Standard VGG architectures (VGG11 to VGG19).
- **ResNet**: Standard ResNet architectures (ResNet34 to ResNet152).
- **SimpleCNN**: A configurable convolutional neural network.
- **LinearNet**: A dense neural network for tabular data.
- **HFTextClassifier**: A decoder LLM (e.g. TinyLlama-1.1B) used as a LoRA sequence classifier for text risks (requires the optional `llm` extra).

### Risks

Amulet provides attacks, defenses, and evaluation metrics for the following risks:

#### Security

- **Evasion**: Projected Gradient Descent (PGD) attacks and Adversarial Training.
- **Poisoning**: BadNets backdoor attacks (image/tabular) and a textual variant (`TextBadNets`) on a LoRA-tuned LLM; Outlier Removal and ONION (inference-time input purification) defenses.
- **Unauthorized Model Ownership**: Model Extraction attacks, Watermarking, and Fingerprinting.

#### Privacy

- **Membership Inference**: Likelihood Ratio Attack (LiRA) and DP-SGD defense.
- **Attribute Inference**: MLP-based inference of sensitive attributes.
- **Distribution Inference**: KL-divergence-based distinguishing tests.
- **Data Reconstruction**: Model Inversion attacks.

#### Fairness

- **Discriminatory Behavior**: Measuring group fairness and Adversarial Debiasing.

## Data Loading

### Data Class

All Amulet datasets are returned as an `AmuletDataset` dataclass:

```python
@dataclass
class AmuletDataset:
    train_set: torch.utils.data.Dataset
    test_set: torch.utils.data.Dataset
    num_features: int
    num_classes: int
    x_train: np.ndarray | None = None
    x_test: np.ndarray | None = None
    y_train: np.ndarray | None = None
    y_test: np.ndarray | None = None
    z_train: np.ndarray | None = None
    z_test: np.ndarray | None = None
```

- `train_set`/`test_set`: PyTorch Datasets ready for use with a `DataLoader`.
- `x_*`/`y_*`: Raw features and labels as NumPy arrays (available for processed datasets like LFW, Census, CelebA).
- `z_*`: Sensitive attributes used by fairness and attribute inference modules.

### Accessing Datasets

The primary entry point for loading data is `load_data`:

```python
from amulet.utils import load_data

data = load_data(
    root="./data",
    dataset="cifar10",       # Options: cifar10, cifar100, fmnist, mnist, census, lfw, celeba, utkface
    training_size=1.0,       # Downsample training data (0.0 to 1.0)
    celeba_target="Smiling", # Target attribute for CelebA
    exp_id=0                 # Random seed for reproducibility
)
```

Text datasets are loaded directly (not through `load_data`), and return an `AmuletDataset` with `modality="text"` whose `train_set`/`test_set` are `TextTensorDataset` instances (padded `input_ids` plus the raw strings). They require the optional `llm` extra:

```python
from amulet.datasets import load_sst2

data = load_sst2(
    path="./data/sst2",                              # project-local HF cache
    tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # tokenizer that produces input_ids
    max_length=128,                                  # fixed sequence length
)
```

## Creating Models

Amulet models subclass `AmuletModel` (itself an `nn.Module`) and implement both `forward(x)` and a `get_hidden(x)` method for accessing intermediate features.

### Initializing Architectures

```python
from amulet.utils import initialize_model

model = initialize_model(
    model_arch="vgg",       # Options: vgg, resnet, linearnet, cnn
    model_capacity="m1",    # Options: m1, m2, m3, m4 (small to large)
    num_features=data.num_features,
    num_classes=data.num_classes,
    batch_norm=True
)
```

## Module Guide

For detailed instructions on each risk, please see the [Module Guide](./module_guide/1_INTRO.md).
Check the [examples/](https://github.com/ssg-research/amulet/tree/main/examples) directory for end-to-end scripts.
