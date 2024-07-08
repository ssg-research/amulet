
# Features

We provide a high-level list of features below. Please refer to the Tutorial (Link TBD) for more information.

## Datasets
Amulet provides the following for computer vision tasks:
- [CIFAR10](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html).
- [FashionMNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html).

Amulet pre-processes and provides the following datasets to test for privacy-related concerns:
- [Census Income Dataset](https://archive.ics.uci.edu/dataset/20/census+income).
- [Labeled Faces in the Wild (LFW)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html).
## Models
Amulet provides the following models:
- [VGG](https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/): for computer vision tasks.
- LinearNet: A dense neural network tuned for multiclass classification on the FashionMNIST dataset.
- BinaryNet: A dense neural network tuned for binary classification.

## Risks
Amulet provides attacks, defenses, and evaluation metrics for the following risks:
### Security
- Evasion
- Poisoning
- Unauthorized Model Ownership

### Privacy
- Membership Inference
- Attribute Inference
- Distribution Inference
- Data Reconstruction

### Fairness
- Discriminatory Behavior

Please check the Module Heirarchy (link TBD) for more details.

# Data Loading

## Data Class
All Amulet datasets are returned as an AmuletDataset, which contains the following objects:
```python
train_set: torch.utils.data.Dataset
test_set: torch.utils.data.Dataset
x_train: np.ndarray | None = None
x_test: np.ndarray | None = None
y_train: np.ndarray | None = None
y_test: np.ndarray | None = None
z_train: np.ndarray | None = None
z_test: np.ndarray | None = None
```
Every dataset will have the `train_set` and `test_set` attributes. For datasets that are loaded manually and processed by Amulet (such as LFW and Census Income Dataset), the individual feature and labels arrays will also be set `[x_train, x_test, y_train, y_test]`, as well as the arrays to store sensitive attributes `[z_train, z_test]`.

## Accessing Datasets
For easy access, Amulet provides the following function to load any dataset as part of a pipeline:
```python
def load_data(
    root: Path | str,
    dataset: str,
    training_size: float,
    log: logging.Logger | None = None,
    exp_id: int = 0,
) -> AmuletDataset:
```
Where:
- `root`: the root directory to store datasets.
- `dataset`: one of `['cifar10', 'fminst', 'census', 'lfw']`.
- `training_size`: used to reduce the size of the training data, useful to test the impact of dataset size on models.
- `log`: for logging.
- `exp_id`: for random seeding, if needed.

Amulet provides the following functions to load datasets:
- ```python
    def load_cifar10(
        path: str | Path = Path("./data/cifar10"),
        transform_train: transforms.Compose | None = None,
        transform_test: transforms.Compose | None = None,
    ) -> AmuletDataset:
   ```

- ```python
    def load_fmnist(
        path: str | Path = Path("./data/fmnist"),
        transform_train: transforms.Compose | None = None,
        transform_test: transforms.Compose | None = None,
    ) -> AmuletDataset:
   ```
- ```python
    def load_census(
        path: str | Path = Path("./data/census"),
        random_seed: int = 7,
        test_size: float = 0.5,
    ) -> AmuletDataset:
  ```

- ```python
    def load_lfw(
        path: str | Path = Path("./data/lfw"),
        target: str = "age",
        attribute_1: str = "race",
        attribute_2: str = "gender",
        test_size: float = 0.3,
        random_seed: int = 7,
    ) -> AmuletDataset:
  ```

# Creating Models
Any PyTorch model of type `torch.nn.Module` can be used with Amulet. For some modules, it is helpful to have a `get_hidden(self, x: torch.Tensor) -> torch.Tensor` method that returns the intermediate layer output from the model, please see [this example](https://github.com/ssg-research/amulet/blob/main/amulet/models/vgg.py#L106).

## Accessing pre-configured model architectures
To use the models configured for running experiments using Amulet, the following function can be used:
```python
def initialize_model(
    model_arch: str,
    model_capacity: str,
    dataset: str,
    log: logging.Logger | None = None,
) -> nn.Module:
```
Where:
- `model_arch`: one of `['vgg', 'linearnet', 'binarynet']`. Each model is optimized for a specific dataset that Amulet provides.
- `model_capacity`: one of `['m1', 'm2', 'm3', 'm4]`. Configures the size and complexity of the model.
- `dataset`: one of `['cifar10', 'fminst', 'census', 'lfw']`. Used to determine the input size for some models.
- `log`: for logging.

# Risks
Amulet provides attacks, defenses, and metrics for each risk. **TBD.**
