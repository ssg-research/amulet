"""Slow-tier smokes: one real cold-start download per custom dataset loader.

These are the only tests where real bytes flow from the network. Everything
else about the loaders (orchestration, cache control flow, array contracts,
helpers) is covered in milliseconds by the mocked-boundary unit tests in
tests/unit/test_{celeba,lfw,utkface}_datasets.py. Run via
`uv run pytest -m slow` (scheduled monthly in CI, or on demand).
"""

from pathlib import Path

import numpy as np
import pytest

from amulet.datasets import load_celeba, load_lfw, load_utkface


def _assert_well_formed_image_bundle(data, z_columns: int) -> None:
    assert data.x_train is not None and data.y_train is not None
    assert data.z_train is not None
    assert data.x_train.dtype == np.float32
    assert data.x_train.shape[1:] == (3, 64, 64)
    assert data.y_train.dtype == np.int64
    assert data.z_train.shape == (len(data.y_train), z_columns)


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_celeba_cold_start_real_download(tmp_path: Path) -> None:
    data = load_celeba(path=tmp_path, target_attribute="Smiling")

    assert (tmp_path / "list_attr_celeba.txt").exists()
    assert (tmp_path / "img_align_celeba").is_dir()
    assert (tmp_path / "celeba_processed__target=Smiling.npz").exists()
    _assert_well_formed_image_bundle(data, z_columns=1)


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_lfw_cold_start_real_download(tmp_path: Path) -> None:
    data = load_lfw(
        path=tmp_path, target="age", attribute_1="race", attribute_2="gender"
    )

    assert (tmp_path / "lfw_attributes.txt").exists()
    assert (tmp_path / "lfw_images.npz").exists()
    assert data.x_train is not None and data.y_train is not None
    assert data.x_train.dtype == np.float32
    assert data.num_features == data.x_train.shape[1]
    assert set(np.asarray(data.y_train)) <= {0, 1}


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_utkface_cold_start_real_download(tmp_path: Path) -> None:
    data = load_utkface(path=tmp_path)

    assert (tmp_path / "UTKFace").is_dir()
    assert (
        tmp_path / "utkface_processed__target=age__attr1=gender__attr2=race.npz"
    ).exists()
    _assert_well_formed_image_bundle(data, z_columns=2)
