"""
Contains methods to load reference datasets for
computer vision applications.
"""

import tarfile
import zipfile
from pathlib import Path

import gdown
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torchvision import datasets

from .__data import AmuletDataset

_CELEBA_IMAGES_GDRIVE_ID = "1aiLLTGVnOnq0Ln9uf3nSuj8JmhrX5rMx"
_CELEBA_ATTRS_GDRIVE_ID = "15HHhEpb0ylQliq8vbdrN6kBlxpc33OA5"
_CELEBA_IMG_SIZE = 64
_CELEBA_CROP_SIZE = 148


def load_cifar10(
    path: str | Path = Path("./data/cifar10"),
    transform_train: transforms.Compose | None = None,
    transform_test: transforms.Compose | None = None,
) -> AmuletDataset:
    """
    Load the CIFAR10 dataset with standard transformations.

    Args:
        path: Directory where the dataset is stored or downloaded.
        transform_train: Transforms applied to training images. Defaults to random crop + flip + ToTensor.
        transform_test: Transforms applied to test images. Defaults to ToTensor.

    Returns:
        AmuletDataset with train_set and test_set populated.
    """

    # Note: We do not normalize the inputs by default to preserve [0,1] range.
    # This is important for compatibility with attack methods (e.g., PGD, MIA).

    if transform_train is None:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    if transform_test is None:
        transform_test = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.CIFAR10(
        root=path, train=True, transform=transform_train, download=True
    )
    test_set = datasets.CIFAR10(
        root=path, train=False, transform=transform_test, download=True
    )

    return AmuletDataset(
        train_set=train_set,
        test_set=test_set,
        num_features=32 * 32,
        num_classes=10,
        modality="image",
    )


def load_cifar100(
    path: str | Path = Path("./data/cifar100"),
    transform_train: transforms.Compose | None = None,
    transform_test: transforms.Compose | None = None,
) -> AmuletDataset:
    """
    Load the CIFAR100 dataset with standard transformations.

    Args:
        path: Directory where the dataset is stored or downloaded.
        transform_train: Transforms applied to training images. Defaults to random crop + flip + color jitter + ToTensor.
        transform_test: Transforms applied to test images. Defaults to ToTensor.

    Returns:
        AmuletDataset with train_set and test_set populated.
    """

    # Note: We do not normalize the inputs by default to preserve [0,1] range.
    # This is important for compatibility with attack methods (e.g., PGD, MIA).

    if transform_train is None:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2),
        ])

    if transform_test is None:
        transform_test = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.CIFAR100(
        root=path, train=True, transform=transform_train, download=True
    )
    test_set = datasets.CIFAR100(
        root=path, train=False, transform=transform_test, download=True
    )

    return AmuletDataset(
        train_set=train_set,
        test_set=test_set,
        num_features=32 * 32,
        num_classes=100,
        modality="image",
    )


def load_fmnist(
    path: str | Path = Path("./data/fmnist"),
    transform_train: transforms.Compose | None = None,
    transform_test: transforms.Compose | None = None,
) -> AmuletDataset:
    """
    Load the FashionMNIST dataset with standard transformations.

    Args:
        path: Directory where the dataset is stored or downloaded.
        transform_train: Transforms applied to training images. Defaults to flip + rotation + crop + ToTensor.
        transform_test: Transforms applied to test images. Defaults to flip + rotation + crop + ToTensor.

    Returns:
        AmuletDataset with train_set and test_set populated.
    """
    if transform_train is None:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomCrop([28, 28]),
            transforms.ToTensor(),
        ])

    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomCrop([28, 28]),
            transforms.ToTensor(),
        ])

    train_set = datasets.FashionMNIST(
        root=path, train=True, transform=transform_train, download=True
    )
    test_set = datasets.FashionMNIST(
        root=path, train=False, transform=transform_test, download=True
    )

    return AmuletDataset(
        train_set=train_set,
        test_set=test_set,
        num_features=28 * 28,
        num_classes=10,
        modality="image",
    )


def load_mnist(
    path: str | Path = Path("./data/mnist"),
    transform_train: transforms.Compose | None = None,
    transform_test: transforms.Compose | None = None,
) -> AmuletDataset:
    """
    Load the MNIST dataset with standard transformations.

    Args:
        path: Directory where the dataset is stored or downloaded.
        transform_train: Transforms applied to training images. Defaults to flip + rotation + crop + ToTensor.
        transform_test: Transforms applied to test images. Defaults to flip + rotation + crop + ToTensor.

    Returns:
        AmuletDataset with train_set and test_set populated.
    """
    if transform_train is None:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomCrop([28, 28]),
            transforms.ToTensor(),
        ])

    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomCrop([28, 28]),
            transforms.ToTensor(),
        ])

    train_set = datasets.MNIST(
        root=path, train=True, transform=transform_train, download=True
    )
    test_set = datasets.MNIST(
        root=path, train=False, transform=transform_test, download=True
    )

    return AmuletDataset(
        train_set=train_set,
        test_set=test_set,
        num_features=28 * 28,
        num_classes=10,
        modality="image",
    )


def _celeba_ensure_raw(path: Path) -> tuple[Path, Path]:
    """Download and extract CelebA raw files if not already present locally."""
    attrs_path = path / "list_attr_celeba.txt"
    imgs_dir = path / "img_align_celeba"
    zip_path = path / "img_align_celeba.zip"

    if not attrs_path.exists():
        print("Downloading CelebA attributes from Google Drive...")
        gdown.download(id=_CELEBA_ATTRS_GDRIVE_ID, output=str(attrs_path), quiet=False)

    if not imgs_dir.exists():
        if not zip_path.exists():
            print("Downloading CelebA images from Google Drive...")
            gdown.download(
                id=_CELEBA_IMAGES_GDRIVE_ID, output=str(zip_path), quiet=False
            )
        print("Extracting CelebA images...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(path)

    return attrs_path, imgs_dir


def _celeba_build_processed_cache(
    attrs_path: Path,
    imgs_dir: Path,
    cache_path: Path,
    target_attribute: str,
) -> None:
    """Crop, resize, and align all CelebA images with attribute labels; save to cache."""
    attr_df = pd.read_csv(
        attrs_path,
        sep=r"\s+",
        header=1,
        index_col=0,
    )
    # Attribute values are -1/1; convert to 0/1
    attr_df = (attr_df + 1) // 2

    y = attr_df[target_attribute].to_numpy(dtype=np.int64)
    z = attr_df["Male"].to_numpy(dtype=np.int64)
    filenames = attr_df.index.tolist()
    n = len(filenames)

    crop = transforms.CenterCrop(_CELEBA_CROP_SIZE)
    resize = transforms.Resize(
        (_CELEBA_IMG_SIZE, _CELEBA_IMG_SIZE),
        antialias=True,  # type: ignore[reportCallIssue]
    )

    imgs = np.zeros((n, 3, _CELEBA_IMG_SIZE, _CELEBA_IMG_SIZE), dtype=np.uint8)
    for i, fname in enumerate(filenames):
        img = resize(crop(Image.open(imgs_dir / fname).convert("RGB")))
        imgs[i] = np.asarray(img).transpose(2, 0, 1)
        if i % 20000 == 0:
            print(f"  {i}/{n} images processed...")

    np.savez(cache_path, imgs=imgs, y=y, z=z)


def load_celeba(
    path: str | Path = Path("./data/celeba"),
    random_seed: int = 0,
    test_size: float = 0.5,
    target_attribute: str = "Smiling",
) -> AmuletDataset:
    """
    Load the CelebA dataset, separating sensitive attributes from features and labels.

    On first use the images and attributes are downloaded from Google Drive, extracted,
    and processed into a parameter-keyed .npz cache. Subsequent calls with the same
    target attribute load directly from the cache. Changing target_attribute reprocesses
    from local raw files without re-downloading.

    Args:
        path: Directory where raw and cached dataset files are stored.
        random_seed: Random seed for reproducible train/test splitting.
        test_size: Proportion of data used for testing.
        target_attribute: Binary attribute to use as the classification target.
            Options include: "Smiling", "Wavy_Hair", "Attractive", "Young".

    Returns:
        AmuletDataset with train_set, test_set, x_*, y_*, and z_* arrays populated.
        Images are float32 in [0, 1] with shape (N, 3, 64, 64). Sensitive attribute
        is always "Male" with shape (N, 1).
    """
    if isinstance(path, str):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    cache_path = path / f"celeba_processed__target={target_attribute}.npz"

    if not cache_path.exists():
        attrs_path, imgs_dir = _celeba_ensure_raw(path)
        print(f"Processing CelebA with target={target_attribute!r}...")
        _celeba_build_processed_cache(
            attrs_path, imgs_dir, cache_path, target_attribute
        )

    npz = np.load(cache_path, mmap_mode="r")
    imgs_all: np.ndarray = npz["imgs"]  # (N, 3, 64, 64) uint8, memory-mapped
    y_all: np.ndarray = np.array(npz["y"])
    z_all: np.ndarray = np.array(npz["z"])

    idx_train, idx_test = train_test_split(
        np.arange(len(y_all)),
        test_size=test_size,
        stratify=y_all,
        random_state=random_seed,
    )

    x_train = imgs_all[idx_train].astype(np.float32) / 255.0
    x_test = imgs_all[idx_test].astype(np.float32) / 255.0
    npz.close()

    y_train = y_all[idx_train]
    y_test = y_all[idx_test]
    z_train = z_all[idx_train].reshape(-1, 1)
    z_test = z_all[idx_test].reshape(-1, 1)

    train_set = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
    )
    test_set = TensorDataset(
        torch.from_numpy(x_test),
        torch.from_numpy(y_test),
    )

    return AmuletDataset(
        train_set=train_set,
        test_set=test_set,
        num_features=_CELEBA_IMG_SIZE * _CELEBA_IMG_SIZE,
        num_classes=2,
        modality="image",
        sensitive_columns=["Male"],
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        z_train=z_train,
        z_test=z_test,
    )


_UTKFACE_GDRIVE_ID = "1bon52kNwTneUisLSFiXXbwoAp4huXD-g"
_UTKFACE_IMG_SIZE = 64


def _utkface_ensure_raw(path: Path) -> Path:
    """Download and extract UTKFace if not present; return the images directory."""
    tar_path = path / "UTKFace.tar.gz"
    imgs_dir = path / "UTKFace"

    if not imgs_dir.exists():
        if not tar_path.exists():
            print("Downloading UTKFace from Google Drive...")
            gdown.download(id=_UTKFACE_GDRIVE_ID, output=str(tar_path), quiet=False)
        print("Extracting UTKFace...")
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(path)
        if not imgs_dir.exists():
            # Tar extracted flat — find parent of first matching image
            matches = list(path.rglob("[0-9]*_[0-9]*_[0-9]*_*.jpg"))
            if not matches:
                raise RuntimeError("UTKFace extraction produced no recognisable images")
            imgs_dir = matches[0].parent

    return imgs_dir


def _utkface_parse_labels(img_path: Path) -> tuple[int, int, int] | None:
    """Parse (age, gender, race) from a UTKFace filename; return None if malformed."""
    parts = img_path.stem.split("_")
    if len(parts) < 3:
        return None
    try:
        age = int(parts[0])
        gender = int(parts[1])
        race = int(parts[2])
    except ValueError:
        return None
    if not (0 <= age <= 116) or gender not in (0, 1) or race not in range(5):
        return None
    return age, gender, race


def _utkface_build_processed_cache(
    imgs_dir: Path,
    cache_path: Path,
    target: str,
    attribute_1: str,
    attribute_2: str,
) -> None:
    """Load, resize all UTKFace images and save a parameter-keyed processed cache."""
    valid_paths: list[Path] = []
    ages: list[int] = []
    genders: list[int] = []
    races: list[int] = []

    for img_path in sorted(imgs_dir.glob("*.jpg")):
        labels = _utkface_parse_labels(img_path)
        if labels is None:
            continue
        age, gender, race = labels
        valid_paths.append(img_path)
        ages.append(age)
        genders.append(gender)
        races.append(race)

    label_map: dict[str, np.ndarray] = {
        "age": np.array(ages, dtype=np.int64),
        "gender": np.array(genders, dtype=np.int64),
        "race": np.array(races, dtype=np.int64),
    }

    n = len(valid_paths)
    resize = transforms.Resize(
        (_UTKFACE_IMG_SIZE, _UTKFACE_IMG_SIZE),
        antialias=True,  # type: ignore[reportCallIssue]
    )
    imgs = np.zeros((n, 3, _UTKFACE_IMG_SIZE, _UTKFACE_IMG_SIZE), dtype=np.uint8)
    for i, img_path in enumerate(valid_paths):
        img = resize(Image.open(img_path).convert("RGB"))
        imgs[i] = np.asarray(img).transpose(2, 0, 1)
        if i % 5000 == 0:
            print(f"  {i}/{n} images processed...")

    np.savez(
        cache_path,
        imgs=imgs,
        y=label_map[target],
        z1=label_map[attribute_1],
        z2=label_map[attribute_2],
    )


def load_utkface(
    path: str | Path = Path("./data/utkface"),
    target: str = "age",
    attribute_1: str = "gender",
    attribute_2: str = "race",
    age_bins: list[int] | None = None,
    test_size: float = 0.3,
    random_seed: int = 7,
) -> AmuletDataset:
    """
    Load the UTKFace dataset with age, gender, and race labels parsed from filenames.

    On first use the archive is downloaded from Google Drive, extracted, and processed
    into a parameter-keyed .npz cache. Subsequent calls with the same target/attributes
    load from the cache. Changing target or attributes reprocesses from local files
    without re-downloading.

    Labels are stored raw: age as integer 0-116, gender as 0 (male) / 1 (female),
    race as 0-4 (White / Black / Asian / Indian / Other). Pass age_bins to discretize
    any age attribute into groups, e.g. age_bins=[30, 60] produces three groups
    (0-29, 30-59, 60+) via numpy.digitize.

    Args:
        path: Directory where raw and cached dataset files are stored.
        target: Attribute to use as the classification target. Options: "age", "gender", "race".
        attribute_1: First sensitive attribute. Options: "age", "gender", "race".
        attribute_2: Second sensitive attribute. Options: "age", "gender", "race".
        age_bins: Bin edges for discretizing age labels. Applied to every attribute
            that is "age". None returns raw integer age.
        test_size: Proportion of data used for testing.
        random_seed: Random seed for reproducible train/test splitting.

    Returns:
        AmuletDataset with train_set, test_set, x_*, y_*, and z_* arrays populated.
        Images are float32 in [0, 1] with shape (N, 3, 64, 64).
        z_train/z_test have shape (N, 2) for the two sensitive attributes.
    """
    if isinstance(path, str):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    cache_path = (
        path
        / f"utkface_processed__target={target}__attr1={attribute_1}__attr2={attribute_2}.npz"
    )

    if not cache_path.exists():
        imgs_dir = _utkface_ensure_raw(path)
        print(
            f"Processing UTKFace with target={target!r}, "
            f"attr1={attribute_1!r}, attr2={attribute_2!r}..."
        )
        _utkface_build_processed_cache(
            imgs_dir, cache_path, target, attribute_1, attribute_2
        )

    npz = np.load(cache_path, mmap_mode="r")
    imgs_all: np.ndarray = npz["imgs"]
    y_all: np.ndarray = np.array(npz["y"])
    z1_all: np.ndarray = np.array(npz["z1"])
    z2_all: np.ndarray = np.array(npz["z2"])
    npz.close()

    # Apply age binning to any attribute that is "age"
    if age_bins is not None:
        bins = np.array(age_bins)
        if target == "age":
            y_all = np.digitize(y_all, bins).astype(np.int64)
        if attribute_1 == "age":
            z1_all = np.digitize(z1_all, bins).astype(np.int64)
        if attribute_2 == "age":
            z2_all = np.digitize(z2_all, bins).astype(np.int64)

    _, counts = np.unique(y_all, return_counts=True)
    stratify = y_all if int(counts.min()) >= 2 else None
    idx_train, idx_test = train_test_split(
        np.arange(len(y_all)),
        test_size=test_size,
        stratify=stratify,
        random_state=random_seed,
    )

    x_train = imgs_all[idx_train].astype(np.float32) / 255.0
    x_test = imgs_all[idx_test].astype(np.float32) / 255.0

    y_train = y_all[idx_train]
    y_test = y_all[idx_test]
    z_train = np.stack([z1_all[idx_train], z2_all[idx_train]], axis=1)
    z_test = np.stack([z1_all[idx_test], z2_all[idx_test]], axis=1)

    train_set = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
    )
    test_set = TensorDataset(
        torch.from_numpy(x_test),
        torch.from_numpy(y_test),
    )

    return AmuletDataset(
        train_set=train_set,
        test_set=test_set,
        num_features=_UTKFACE_IMG_SIZE * _UTKFACE_IMG_SIZE,
        num_classes=len(np.unique(y_all)),
        modality="image",
        sensitive_columns=[attribute_1, attribute_2],
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        z_train=z_train,
        z_test=z_test,
    )
