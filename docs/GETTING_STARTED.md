# Getting Started

Scarab is a PyTorch-based research library for evaluating unintended interactions among machine learning (ML) defenses and risks across security, privacy, and fairness.

## Features

### Datasets

Scarab provides built-in support for several common datasets, including automated downloading and pre-processing:

- **Computer Vision**: CIFAR-10, CIFAR-100, FashionMNIST, MNIST.
- **Face Attributes**: CelebA, Labeled Faces in the Wild (LFW), UTKFace.
- **Tabular Data**: Census Income Dataset.
- **Text**: SST-2, AG News, IMDB (loaded via Hugging Face `datasets`; requires the optional `llm` extra).

### Models

Scarab provides pre-configured architectures with scalable capacity:

- **VGG**: Standard VGG architectures (VGG11 to VGG19).
- **ResNet**: Standard ResNet architectures (ResNet34 to ResNet152).
- **SimpleCNN**: A configurable convolutional neural network.
- **LinearNet**: A dense neural network for tabular data.
- **HFCausalLM**: A HuggingFace causal (decoder-only) LM (e.g. TinyLlama-1.1B) adapted with LoRA, usable as a classifier, a perplexity scorer, or a generator for text risks (requires the optional `llm` extra).

### Risks

Scarab provides attacks, defenses, and evaluation metrics for the following risks:

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

All Scarab datasets are returned as a `ScarabDataset` dataclass:

```python
@dataclass
class ScarabDataset:
    train_set: torch.utils.data.Dataset
    test_set: torch.utils.data.Dataset
    num_features: int
    num_classes: int
    modality: Literal["image", "tabular", "text"]
    sensitive_columns: list[str] | None = None
    x_train: np.ndarray | None = None
    x_test: np.ndarray | None = None
    y_train: np.ndarray | None = None
    y_test: np.ndarray | None = None
    z_train: np.ndarray | None = None
    z_test: np.ndarray | None = None
```

- `train_set`/`test_set`: PyTorch Datasets ready for use with a `DataLoader`.
- `modality`: Required. One of `"image"`, `"tabular"`, or `"text"`; it tells models the shape of each sample.
- `sensitive_columns`: Column names of `z_train`/`z_test` in order, or `None` if the dataset has no sensitive attributes.
- `x_*`/`y_*`: Raw features and labels as NumPy arrays (available for processed datasets like LFW, Census, CelebA).
- `z_*`: Sensitive attributes used by fairness and attribute inference modules.

### Accessing Datasets

The primary entry point for loading data is `load_data`:

```python
from scarab.utils import load_data

data = load_data(
    root="./data",
    dataset="cifar10",       # Options: cifar10, cifar100, fmnist, mnist, census, lfw, celeba, utkface
    training_size=1.0,       # Downsample training data (0.0 to 1.0)
    celeba_target="Smiling", # Target attribute for CelebA
    exp_id=0                 # Random seed for reproducibility
)
```

Text datasets are loaded directly (not through `load_data`), and return a `ScarabDataset` with `modality="text"` whose `train_set`/`test_set` are `TextTensorDataset` instances (padded `input_ids` plus the raw strings). They require the optional `llm` extra:

```python
from scarab.datasets import load_sst2

data = load_sst2(
    path="./data/sst2",                              # project-local HF cache
    tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # tokenizer that produces input_ids
    max_length=128,                                  # fixed sequence length
)
```

## Creating Models

Scarab models subclass `ScarabModel` (itself an `nn.Module`) and implement both `forward(x)` and a `get_hidden(x)` method for accessing intermediate features.

### Initializing Architectures

```python
from scarab.utils import initialize_model

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
Check the [examples/](../examples/) directory for end-to-end scripts.
