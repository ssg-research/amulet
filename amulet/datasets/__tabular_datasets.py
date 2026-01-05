"""
Contains methods to load reference tabular datasets that are
used in applications with sensitive data attributes.
"""

import os
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ucimlrepo import fetch_ucirepo
from PIL import Image

from .__data import AmuletDataset


def load_census(
    path: str | Path = Path("./data/census"),
    random_seed: int = 7,
    test_size: float = 0.5,
) -> AmuletDataset:
    """
    Loads the Census Income dataset from https://archive.ics.uci.edu/dataset/20/census+income.
    Applies data standard data cleaning and one-hot encoding. Separates the sensitive attributes
    from the training and testing data.

    Args:
        path: str or Path object
            String or Path object indicating where to store the dataset.
        random_seed: int
            Determines random number generation for dataset shuffling. Pass an int
            for reproducible output across multiple function calls.
        test_size: float
            Proportion of data used for testing.
    Returns:
        Object (:class:`~amulet.datasets.Data`), with the following attributes:
            train_set: :class:`~torch.utils.data.VisionDataset`
                A dataset of images and labels used to build a DataLoader for
                training PyTorch models.
            test_set: :class:`~torch.utils.data.VisionDataset`
                A dataset of images and labels used to build a DataLoader for
                testing PyTorch models.
            x_train: :class:`~np.ndarray`
                Features for the train data.
            x_test: :class:`~np.ndarray`
                Features for the test data.
            y_train: :class:`~np.ndarray`
                Labels for the train data.
            y_test: :class:`~np.ndarray`
                Labels for the test data.
            z_train: :class:`~np.ndarray`
                Sensitive attribute labels for the train data.
            z_test: :class:`~np.ndarray`
                Sensitive attribute labels for the test data.
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
    column_names = list(dtypes.keys())
    if isinstance(path, str):
        path = Path(path)

    filename = path / "adult.csv"
    if not os.path.exists(filename):
        path.mkdir(parents=True, exist_ok=True)
        # Type error in ucimlrepo library
        adult_data = fetch_ucirepo(id=2).data.original[column_names]  # type: ignore[reportOptionalMemberAccess]
        adult_data = adult_data.replace("?", np.NaN).loc[
            lambda df: df["race"].isin(["White", "Black"])
        ]
        adult_data.to_csv(filename, index=False)
    else:
        adult_data = pd.read_csv(filename, dtype=dtypes)  # type: ignore[reportCallIssue, reportArgumentType]

    # Split data into features / sensitive features / target
    sensitive_attributes = ["race", "sex"]
    sensitive_features = adult_data.loc[:, sensitive_attributes].assign(
        race=lambda df: (df["race"] == "White").astype(int),
        sex=lambda df: (df["sex"] == "Male").astype(int),
    )
    target = (adult_data["income"] == ">50K").astype(int)
    to_drop = ["income", "fnlwgt"] + sensitive_attributes
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
        x_train=np.array(x_train),
        x_test=np.array(x_test),
        y_train=np.array(y_train),
        y_test=np.array(y_test),
        z_train=np.array(z_train),
        z_test=np.array(z_test),
    )


def load_lfw(
    path: str | Path = Path("./data/lfw"),
    target: str = "age",
    attribute_1: str = "race",
    attribute_2: str = "gender",
    test_size: float = 0.3,
    random_seed: int = 7,
) -> AmuletDataset:
    """
    Loads the Labeled Faces in the Wild (LFW) Dataset from Scikit-Learn and
    combines it with attributes for each image from
    https://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt
    to convert it to a Property Inference problem. Sample code taken from
    https://github.com/csong27/property-inference-collaborative-ml/blob/master/load_lfw.py

    Reference:
        L. Melis, C. Song, E. De Cristofaro and V. Shmatikov,
        "Exploiting Unintended Feature Leakage in Collaborative Learning",
        2019 IEEE Symposium on Security and Privacy (SP),
        San Francisco, CA, USA, 2019, pp. 691-706, doi: 10.1109/SP.2019.00029.

    Args:
        path: str or Path object
            String or Path object indicating where to store the dataset.
        target: str
            Defines which attribute to use as a target, possible values:
            [age, gender, race]
        attribute_1: str
            Defines which attribute to use as a sensitive attribute, possible values:
            [age, gender, race]
        attribute_2: str
            Defines which attribute to use as a sensitive attribute, possible values:
            [age, gender, race]
        test_size: float
            Proportion of data used for testing.
        random_seed: int
            Determines random number generation for dataset shuffling. Pass an int
            for reproducible output across multiple function calls.
    Returns:
        Object (:class:`~amulet.datasets.Data`), with the following attributes:
            train_set: :class:`~torch.utils.data.VisionDataset`
                A dataset of images and labels used to build a DataLoader for
                training PyTorch models.
            test_set: :class:`~torch.utils.data.VisionDataset`
                A dataset of images and labels used to build a DataLoader for
                testing PyTorch models.
            x_train: :class:`~np.ndarray`
                Features for the train data.
            x_test: :class:`~np.ndarray`
                Features for the test data.
            y_train: :class:`~np.ndarray`
                Labels for the train data.
            y_test: :class:`~np.ndarray`
                Labels for the test data.
            z_train: :class:`~np.ndarray`
                Sensitive attribute labels for the train data.
            z_test: :class:`~np.ndarray`
                Sensitive attribute labels for the test data.
    """
    if isinstance(path, str):
        path = Path(path)

    attributes_path = path / "lfw_attributes.txt"
    images_path = path / "lfw_images.npz"
    if not os.path.exists(images_path):
        # Download and save data
        print("Downloading LFW Data")
        fetch_lfw_people(color=True, data_home=path)
        with urlopen(
            "https://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt"
        ) as response:
            attributes_file = response.read().decode("utf-8")
            attributes_file = "\n".join(attributes_file.split("\n")[1:]).replace(
                "#\t", ""
            )
            with open(attributes_path, "w", encoding="utf-8") as f:
                f.write(attributes_file)

        attributes = pd.read_csv(attributes_path, delimiter="\t")

        names = np.asarray(attributes["person"])
        img_num = np.asarray(attributes["imagenum"])

        slice_ = (slice(70, 195), slice(78, 172))
        h_slice, w_slice = slice_
        h = (h_slice.stop - h_slice.start) // (h_slice.step or 1)
        w = (w_slice.stop - w_slice.start) // (w_slice.step or 1)

        resize_by = 0.5
        h = int(resize_by * h)
        w = int(resize_by * w)

        imgs = np.zeros((len(names), h, w, 3), dtype=np.uint8)
        i = 0
        for name, num in zip(names, img_num):
            name = name.replace(" ", "_")
            img_path = (
                path
                / "lfw_home"
                / "lfw_funneled"
                / name
                / f"{name}_{str(num).zfill(4)}.jpg"
            )
            img = Image.fromarray(np.array(Image.open(img_path))[slice_])
            img = np.array(img.resize((w, h)))
            imgs[i] = img
            i += 1

        np.savez(images_path, imgs)

    BINARY_ATTRS = {"gender": ["Female", "Male"], "smile": ["Not Smiling", "Smiling"]}

    MULTI_ATTRS = {
        "race": ["White", "Black"],
        # 'race': ['Asian', 'White', 'Black'], TODO: For most algorithms we require binary attributes. Need to generalize.
        "glasses": ["Eyeglasses", "Sunglasses", "No Eyewear"],
        "age": ["Baby", "Child", "Youth", "Middle Aged", "Senior"],
        "hair": ["Black Hair", "Blond Hair", "Brown Hair", "Bald"],
    }

    attributes = pd.read_csv(attributes_path, delimiter="\t", low_memory=False)

    # Function to load binary attributes
    def _load_lfw_binary_attr(attribute: str):
        if attribute == "gender":
            binary_attr = np.asarray(attributes["Male"])
            binary_attr = np.sign(binary_attr)
            binary_attr[binary_attr == -1] = 0
            return dict(zip(range(len(binary_attr)), binary_attr))
        else:
            raise ValueError(attribute)

    # Function to load multi-class attributes
    def _load_lfw_multi_attr(attr_type: str, thresh: float = -0.1):
        if attr_type == "race" or attr_type == "age":
            multi_attrs = np.asarray(attributes[MULTI_ATTRS[attr_type]])
        else:
            raise ValueError(attr_type)

        indices = []
        labels = []
        for i, a in enumerate(multi_attrs):
            if np.max(a) < thresh:  # The score is too low for an attribute
                continue
            indices.append(i)
            labels.append(np.argmax(a))

        return dict(zip(indices, labels))

    # Wrapper
    def _load_lfw_attr(attribute: str):
        return (
            _load_lfw_binary_attr(attribute)
            if attribute in BINARY_ATTRS
            else _load_lfw_multi_attr(attribute)
        )

    # Load images and labels
    with np.load(images_path) as f:
        imgs = f["arr_0"].transpose(0, 3, 1, 2)

    target_labels = _load_lfw_attr(target)
    target_indices = [x for x in target_labels.keys()]

    sensitive_attr_1 = _load_lfw_attr(attribute_1)
    sens_1_indices = [x for x in sensitive_attr_1.keys()]

    sensitive_attr_2 = _load_lfw_attr(attribute_2)
    sens_2_indices = [x for x in sensitive_attr_2.keys()]

    # Align and normalize data
    common_indices = np.intersect1d(target_indices, sens_1_indices)
    common_indices = np.intersect1d(common_indices, sens_2_indices)
    imgs = imgs[common_indices] / np.float32(255.0)
    target_array = np.asarray(
        [target_labels[i] for i in common_indices], dtype=np.int32
    )
    sensitive_attrs_1 = np.asarray(
        [sensitive_attr_1[i] for i in common_indices], dtype=np.int32
    )
    sensitive_attrs_2 = np.asarray(
        [sensitive_attr_2[i] for i in common_indices], dtype=np.int32
    )
    x = imgs.reshape((imgs.shape[0], -1))

    index = ["Row" + str(num) for num in range(x.shape[0])]
    columns = ["Col" + str(num) for num in range(x.shape[1])]

    # Convert to DataFrames
    x = pd.DataFrame(data=x, index=index, columns=columns)  # type: ignore[reportArgumentType]
    y = pd.DataFrame({target: target_array})
    if attribute_1 == "gender":
        attribute_1 = "sex"
    elif attribute_2 == "gender":
        attribute_2 = "sex"
    z = pd.DataFrame({attribute_1: sensitive_attrs_1, attribute_2: sensitive_attrs_2})

    # Convert target to binary classification task
    y["age"] = y["age"].replace(1, 0)
    y["age"] = y["age"].replace(2, 0)
    y["age"] = y["age"].replace(3, 1)
    y["age"] = y["age"].replace(4, 1)

    # Split data into train / test
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
        x, y, z, test_size=test_size, random_state=random_seed
    )

    # Create datasets
    train_set = TensorDataset(
        torch.from_numpy(np.array(x_train)).type(torch.float),
        torch.from_numpy(np.array(y_train)).type(torch.long).squeeze(1),
    )
    test_set = TensorDataset(
        torch.from_numpy(np.array(x_test)).type(torch.float),
        torch.from_numpy(np.array(y_test)).type(torch.long).squeeze(1),
    )

    y_train, y_test = np.array(y_train).reshape(-1), np.array(y_test).reshape(-1)
    return AmuletDataset(
        train_set=train_set,
        test_set=test_set,
        num_features=8742,
        num_classes=2,
        x_train=np.array(x_train),
        x_test=np.array(x_test),
        y_train=y_train,
        y_test=y_test,
        z_train=np.array(z_train),
        z_test=np.array(z_test),
    )
