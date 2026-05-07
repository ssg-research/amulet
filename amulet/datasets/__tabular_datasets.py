"""
Contains methods to load reference tabular datasets that are
used in applications with sensitive data attributes.
"""

import io
from pathlib import Path

import gdown
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset

from .__data import AmuletDataset

_CENSUS_GDRIVE_ID = "1sk8q1DElWbeNfVq1Gi0mdqTyUf-I2vIS"

_LFW_ATTRIBUTES_GDRIVE_ID = "1jQZFXZwRxqZ-AwVQukX4I3UHUHMEW7Lu"

_LFW_BINARY_ATTRS: dict[str, list[str]] = {
    "gender": ["Female", "Male"],
    "smile": ["Not Smiling", "Smiling"],
}

_LFW_MULTI_ATTRS: dict[str, list[str]] = {
    "race": ["White", "Black"],
    "glasses": ["Eyeglasses", "Sunglasses", "No Eyewear"],
    "age": ["Baby", "Child", "Youth", "Middle Aged", "Senior"],
    "hair": ["Black Hair", "Blond Hair", "Brown Hair", "Bald"],
}


def load_census(
    path: str | Path = Path("./data/census"),
    random_seed: int = 7,
    test_size: float = 0.5,
) -> AmuletDataset:
    """
    Load the Census Income dataset with cleaning, one-hot encoding, and sensitive attribute separation.

    Args:
        path: Directory where the dataset CSV is stored or downloaded.
        random_seed: Random seed for reproducible train/test splitting.
        test_size: Proportion of data used for testing.

    Returns:
        AmuletDataset with train_set, test_set, x_*, y_*, and z_* arrays populated.
    """
    dtypes = {
        "age": int,
        "workclass": str,
        "fnlwgt": int,
        "education": str,
        "education-num": int,
        "marital-status": str,
        "occupation": str,
        "relationship": str,
        "race": str,
        "sex": str,
        "capital-gain": int,
        "capital-loss": int,
        "hours-per-week": int,
        "native-country": str,
        "income": str,
    }
    if isinstance(path, str):
        path = Path(path)

    filename = path / "adult.csv"
    if not filename.exists():
        path.mkdir(parents=True, exist_ok=True)
        print("Downloading Census data from Google Drive...")
        gdown.download(id=_CENSUS_GDRIVE_ID, output=str(filename), quiet=False)

    adult_data = pd.read_csv(filename, dtype=dtypes)  # type: ignore[reportCallIssue, reportArgumentType]

    # Split data into features / sensitive features / target
    sensitive_attributes = ["race", "sex"]
    sensitive_features = adult_data.loc[:, sensitive_attributes].assign(
        race=lambda df: (df["race"] == "White").astype(int),
        sex=lambda df: (df["sex"] == "Male").astype(int),
    )
    target = (adult_data["income"] == ">50K").astype(int)
    to_drop = ["income", "fnlwgt", *sensitive_attributes]
    features = (
        adult_data.drop(columns=to_drop)
        .fillna("Unknown")
        .pipe(pd.get_dummies, drop_first=True)
    )

    # Split data into train / test
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
        features,
        target,
        sensitive_features,
        test_size=test_size,
        stratify=target,
        random_state=random_seed,
    )

    # Normalize data to 0 and 1
    int_cols = x_train.select_dtypes(include="int64").columns  # type: ignore[reportAttributeAccessIssue]
    scaler = MinMaxScaler().fit(x_train[int_cols])

    x_train[int_cols] = scaler.transform(x_train[int_cols])
    x_test[int_cols] = scaler.transform(x_test[int_cols])

    bool_cols = x_train.select_dtypes(include="bool").columns  # type: ignore[reportAttributeAccessIssue]
    x_train[bool_cols] = x_train[bool_cols].astype("float")
    x_test[bool_cols] = x_test[bool_cols].astype("float")

    # Create datasets
    train_set = TensorDataset(
        torch.from_numpy(np.array(x_train)).type(torch.float),
        torch.from_numpy(np.array(y_train)).type(torch.long),
    )
    test_set = TensorDataset(
        torch.from_numpy(np.array(x_test)).type(torch.float),
        torch.from_numpy(np.array(y_test)).type(torch.long),
    )

    return AmuletDataset(
        train_set=train_set,
        test_set=test_set,
        num_features=93,
        num_classes=2,
        modality="tabular",
        sensitive_columns=sensitive_attributes,
        x_train=np.array(x_train),
        x_test=np.array(x_test),
        y_train=np.array(y_train),
        y_test=np.array(y_test),
        z_train=np.array(z_train),
        z_test=np.array(z_test),
    )


def _lfw_read_attributes(attributes_path: Path) -> pd.DataFrame:
    """Read lfw_attributes.txt, handling both raw (two-line header) and pre-cleaned formats."""
    with open(attributes_path, encoding="utf-8") as f:
        lines = f.readlines()
    # Raw file has a description comment on line 1 and a "#\t"-prefixed header on line 2.
    if lines[0].startswith("# LFW"):
        lines = lines[1:]
    if lines[0].startswith("#\t"):
        lines[0] = lines[0][2:]
    return pd.read_csv(io.StringIO("".join(lines)), delimiter="\t", low_memory=False)


def _lfw_attr_labels(attributes: pd.DataFrame, attribute: str) -> dict[int, int]:
    """Map row index to integer label for one LFW attribute."""
    if attribute in _LFW_BINARY_ATTRS:
        if attribute != "gender":
            raise ValueError(f"Binary attribute not yet implemented: {attribute!r}")
        raw = np.sign(np.asarray(attributes["Male"])).astype(int)
        raw[raw == -1] = 0
        return dict(enumerate(raw.tolist()))
    if attribute in _LFW_MULTI_ATTRS:
        cols = _LFW_MULTI_ATTRS[attribute]
        multi = np.asarray(attributes[cols])
        thresh = -0.1
        return {
            i: int(np.argmax(row))
            for i, row in enumerate(multi)
            if float(np.max(row)) >= thresh
        }
    raise ValueError(f"Unknown LFW attribute: {attribute!r}")


def _lfw_build_images_npz(path: Path, attributes_path: Path, images_path: Path) -> None:
    """Crop and resize all LFW images from lfw_home/ into a single .npz array."""
    attributes = _lfw_read_attributes(attributes_path)
    names = np.asarray(attributes["person"])
    img_nums = np.asarray(attributes["imagenum"])

    h_slice, w_slice = slice(70, 195), slice(78, 172)
    h = int(0.5 * (h_slice.stop - h_slice.start))
    w = int(0.5 * (w_slice.stop - w_slice.start))
    crop = (h_slice, w_slice)

    imgs = np.zeros((len(names), h, w, 3), dtype=np.uint8)
    for i, (name, num) in enumerate(zip(names, img_nums, strict=True)):
        name = name.replace(" ", "_")
        img_path = (
            path
            / "lfw_home"
            / "lfw_funneled"
            / name
            / f"{name}_{str(num).zfill(4)}.jpg"
        )
        arr = np.array(Image.open(img_path))[crop]
        imgs[i] = np.array(Image.fromarray(arr).resize((w, h)))

    np.savez(images_path, arr_0=imgs)


def _lfw_build_processed_cache(
    images_path: Path,
    attributes_path: Path,
    cache_path: Path,
    target: str,
    attribute_1: str,
    attribute_2: str,
) -> None:
    """Align images with attribute labels and save a parameter-keyed processed cache."""
    attributes = _lfw_read_attributes(attributes_path)

    target_labels = _lfw_attr_labels(attributes, target)
    attr1_labels = _lfw_attr_labels(attributes, attribute_1)
    attr2_labels = _lfw_attr_labels(attributes, attribute_2)

    common = np.intersect1d(
        np.intersect1d(list(target_labels), list(attr1_labels)),
        list(attr2_labels),
    )

    with np.load(images_path) as f:
        imgs = f["arr_0"][common].transpose(0, 3, 1, 2) / np.float32(255.0)

    y = np.asarray([target_labels[int(i)] for i in common], dtype=np.int64)
    # Binarize age: Baby/Child/Youth -> 0, Middle Aged/Senior -> 1
    if target == "age":
        y = np.where(y <= 2, 0, 1).astype(np.int64)

    x = imgs.reshape(imgs.shape[0], -1)
    z1 = np.asarray([attr1_labels[int(i)] for i in common], dtype=np.int64)
    z2 = np.asarray([attr2_labels[int(i)] for i in common], dtype=np.int64)

    np.savez(cache_path, x=x, y=y, z1=z1, z2=z2)


def load_lfw(
    path: str | Path = Path("./data/lfw"),
    target: str = "age",
    attribute_1: str = "race",
    attribute_2: str = "gender",
    test_size: float = 0.3,
    random_seed: int = 7,
) -> AmuletDataset:
    """
    Load the LFW dataset combined with face attributes for property inference.

    Combines Scikit-Learn's LFW images with attribute annotations from the PubFig
    dataset to enable distribution inference experiments.

    On first use the attributes file is downloaded from Google Drive and images are
    fetched via scikit-learn. Processed arrays are cached in a parameter-keyed .npz
    file so subsequent calls with the same target/attributes skip all processing.
    When any of those parameters change, only the fast processing step re-runs; raw
    downloads are reused from disk.

    Args:
        path: Directory where raw and cached dataset files are stored.
        target: Attribute to use as classification target. Options: "age", "gender", "race".
        attribute_1: First sensitive attribute. Options: "age", "gender", "race".
        attribute_2: Second sensitive attribute. Options: "age", "gender", "race".
        test_size: Proportion of data used for testing.
        random_seed: Random seed for reproducible train/test splitting.

    Returns:
        AmuletDataset with train_set, test_set, x_*, y_*, and z_* arrays populated.
    """
    if isinstance(path, str):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    attributes_path = path / "lfw_attributes.txt"
    images_path = path / "lfw_images.npz"
    cache_path = (
        path
        / f"lfw_processed__target={target}__attr1={attribute_1}__attr2={attribute_2}.npz"
    )

    if not cache_path.exists():
        # Ensure raw attributes file
        if not attributes_path.exists():
            print("Downloading LFW attributes from Google Drive...")
            gdown.download(
                id=_LFW_ATTRIBUTES_GDRIVE_ID,
                output=str(attributes_path),
                quiet=False,
            )

        # Ensure intermediate image cache
        if not images_path.exists():
            print("Downloading LFW images via scikit-learn...")
            fetch_lfw_people(color=True, data_home=path)
            print("Building LFW image cache...")
            _lfw_build_images_npz(path, attributes_path, images_path)

        print(
            f"Processing LFW with target={target!r}, attr1={attribute_1!r}, attr2={attribute_2!r}..."
        )
        _lfw_build_processed_cache(
            images_path, attributes_path, cache_path, target, attribute_1, attribute_2
        )

    with np.load(cache_path) as f:
        x = f["x"]
        y = f["y"].astype(np.int64)
        z1 = f["z1"].astype(np.int64)
        z2 = f["z2"].astype(np.int64)

    x_df = pd.DataFrame(
        data=x,
        index=[f"Row{i}" for i in range(x.shape[0])],  # type: ignore[reportArgumentType]
        columns=[f"Col{j}" for j in range(x.shape[1])],  # type: ignore[reportArgumentType]
    )
    y_df = pd.DataFrame({target: y})
    z_df = pd.DataFrame({attribute_1: z1, attribute_2: z2})

    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
        x_df, y_df, z_df, test_size=test_size, random_state=random_seed
    )

    train_set = TensorDataset(
        torch.from_numpy(np.array(x_train)).float(),
        torch.from_numpy(np.array(y_train)).long().squeeze(1),
    )
    test_set = TensorDataset(
        torch.from_numpy(np.array(x_test)).float(),
        torch.from_numpy(np.array(y_test)).long().squeeze(1),
    )

    return AmuletDataset(
        train_set=train_set,
        test_set=test_set,
        num_features=x.shape[1],
        num_classes=int(np.max(y)) + 1,
        modality="tabular",
        sensitive_columns=[attribute_1, attribute_2],
        x_train=np.array(x_train),
        x_test=np.array(x_test),
        y_train=np.array(y_train).reshape(-1),
        y_test=np.array(y_test).reshape(-1),
        z_train=np.array(z_train),
        z_test=np.array(z_test),
    )
