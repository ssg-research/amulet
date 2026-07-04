"""Fast tests for load_celeba and its helpers in amulet/datasets/__image_datasets.py.

The network boundary (gdown) is mocked: "downloading" plants tiny synthetic raw
files, so cold-start orchestration, cache-hit, and changed-target control flow
run in milliseconds against the real extraction and cache-building code. The
real download is exercised by the slow-tier smoke in
tests/slow/test_dataset_downloads.py.
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from amulet.datasets.__image_datasets import (  # type: ignore[reportPrivateImportUsage]
    _celeba_build_processed_cache,
    load_celeba,
)

_N_IMAGES = 8


def _smiling(i: int) -> int:
    return 1 if i % 2 == 0 else -1


def _male(i: int) -> int:
    return 1 if i % 3 == 0 else -1


def _young(i: int) -> int:
    return 1 if i % 4 < 2 else -1


def _fake_attrs_text(n: int = _N_IMAGES) -> str:
    """list_attr_celeba.txt format: count line, header line, then rows of
    filename + -1/1 attribute values. Smiling and Young are 4/4 balanced so a
    stratified 50% split has at least two of each class per side."""
    rows = [
        f"{i:06d}.jpg {_smiling(i)} {_male(i)} {_young(i)}" for i in range(1, n + 1)
    ]
    return f"{n}\nSmiling Male Young\n" + "\n".join(rows) + "\n"


def _fake_jpeg_bytes(rng: np.random.Generator) -> bytes:
    buf = io.BytesIO()
    img = Image.fromarray(rng.integers(0, 256, (10, 10, 3), dtype=np.uint8), mode="RGB")
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def make_fake_celeba(tmp_path: Path):
    """Plant the raw CelebA layout (attrs file + images dir) directly on disk."""

    def _make(n: int = _N_IMAGES) -> tuple[Path, Path]:
        imgs_dir = tmp_path / "img_align_celeba"
        imgs_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        for i in range(1, n + 1):
            _ = (imgs_dir / f"{i:06d}.jpg").write_bytes(_fake_jpeg_bytes(rng))
        attrs_path = tmp_path / "list_attr_celeba.txt"
        _ = attrs_path.write_text(_fake_attrs_text(n), encoding="utf-8")
        return attrs_path, imgs_dir

    return _make


@pytest.fixture
def mock_gdown(mocker):
    """Patch gdown.download so 'downloading' writes synthetic raw files instead."""

    def _download(id: str, output: str, quiet: bool = False) -> None:
        out = Path(output)
        if out.name == "list_attr_celeba.txt":
            _ = out.write_text(_fake_attrs_text(), encoding="utf-8")
        elif out.name == "img_align_celeba.zip":
            rng = np.random.default_rng(0)
            with zipfile.ZipFile(out, "w") as zf:
                for i in range(1, _N_IMAGES + 1):
                    zf.writestr(f"img_align_celeba/{i:06d}.jpg", _fake_jpeg_bytes(rng))
        else:
            raise AssertionError(f"unexpected gdown target: {out}")

    return mocker.patch("gdown.download", side_effect=_download)


def test_cold_start_downloads_raw_and_builds_cache(tmp_path: Path, mock_gdown) -> None:
    load_celeba(path=tmp_path, target_attribute="Smiling")

    assert mock_gdown.call_count == 2
    assert (tmp_path / "list_attr_celeba.txt").exists()
    assert (tmp_path / "img_align_celeba").is_dir()
    assert (tmp_path / "celeba_processed__target=Smiling.npz").exists()


def test_cold_start_returns_well_formed_bundle(tmp_path: Path, mock_gdown) -> None:
    data = load_celeba(path=tmp_path, target_attribute="Smiling")

    assert data.x_train is not None and data.x_test is not None
    assert data.y_train is not None and data.y_test is not None
    assert data.z_train is not None
    n_train, n_test = len(data.y_train), len(data.y_test)
    assert n_train + n_test == _N_IMAGES
    assert data.x_train.dtype == np.float32
    assert data.x_train.shape[1:] == (3, 64, 64)
    assert data.x_train.min() >= 0.0 and data.x_train.max() <= 1.0
    assert data.y_train.dtype == np.int64
    assert set(np.concatenate([data.y_train, data.y_test])) <= {0, 1}
    assert data.z_train.shape == (n_train, 1)
    assert data.z_train.dtype == np.int64
    assert data.num_classes == 2


def test_cache_hit_skips_download_and_reproduces_split(
    tmp_path: Path, mock_gdown
) -> None:
    first = load_celeba(path=tmp_path, target_attribute="Smiling")
    mock_gdown.reset_mock()

    second = load_celeba(path=tmp_path, target_attribute="Smiling")

    mock_gdown.assert_not_called()
    assert first.y_train is not None and second.y_train is not None
    np.testing.assert_array_equal(first.y_train, second.y_train)


def test_changed_target_rebuilds_cache_without_redownload(
    tmp_path: Path, mock_gdown
) -> None:
    load_celeba(path=tmp_path, target_attribute="Smiling")
    mock_gdown.reset_mock()

    data = load_celeba(path=tmp_path, target_attribute="Young")

    mock_gdown.assert_not_called()
    assert (tmp_path / "celeba_processed__target=Young.npz").exists()
    assert (tmp_path / "celeba_processed__target=Smiling.npz").exists()
    assert data.y_train is not None and data.y_test is not None
    assert set(np.concatenate([data.y_train, data.y_test])) <= {0, 1}


def test_build_processed_cache_writes_well_formed_arrays(
    tmp_path: Path, make_fake_celeba
) -> None:
    attrs_path, imgs_dir = make_fake_celeba()
    cache_path = tmp_path / "cache.npz"

    _celeba_build_processed_cache(attrs_path, imgs_dir, cache_path, "Smiling")

    npz = np.load(cache_path)
    assert set(npz.files) == {"imgs", "y", "z"}
    assert npz["imgs"].shape == (_N_IMAGES, 3, 64, 64)
    assert npz["imgs"].dtype == np.uint8
    assert npz["y"].shape == (_N_IMAGES,) and npz["y"].dtype == np.int64
    assert npz["z"].shape == (_N_IMAGES,) and npz["z"].dtype == np.int64


def test_build_processed_cache_maps_attrs_to_binary_labels(
    tmp_path: Path, make_fake_celeba
) -> None:
    attrs_path, imgs_dir = make_fake_celeba()
    cache_path = tmp_path / "cache.npz"

    _celeba_build_processed_cache(attrs_path, imgs_dir, cache_path, "Smiling")

    npz = np.load(cache_path)
    # The raw -1/1 attribute values must map exactly to 0/1.
    expected_y = np.array([
        1 if _smiling(i) == 1 else 0 for i in range(1, _N_IMAGES + 1)
    ])
    expected_z = np.array([1 if _male(i) == 1 else 0 for i in range(1, _N_IMAGES + 1)])
    np.testing.assert_array_equal(npz["y"], expected_y)
    np.testing.assert_array_equal(npz["z"], expected_z)
