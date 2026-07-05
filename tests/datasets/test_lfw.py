"""Fast tests for load_lfw and its helpers in amulet/datasets/__tabular_datasets.py.

Both network boundaries are mocked: gdown plants a synthetic attributes file
and the sklearn fetch plants a tiny funneled image tree, so cold-start
orchestration, cache-hit, and changed-params control flow run in milliseconds
against the real image-cropping and cache-building code. The real download is
exercised by the slow-tier smoke in test_downloads.py.

fetch_lfw_people is imported by name inside the loader module, so the patch
targets amulet's module namespace, not sklearn's.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from amulet.datasets.__tabular_datasets import (  # type: ignore[reportPrivateImportUsage]
    _lfw_attr_labels,
    _lfw_read_attributes,
    load_lfw,
)

_AGE_COLS = ["Baby", "Child", "Youth", "Middle Aged", "Senior"]

# (person, imagenum, male, white, black, age_idx). Every row scores >= -0.1 on
# some race and some age column, so the label-dict intersection keeps all rows.
# Age indices 0-2 binarize to y=0 and 3-4 to y=1, four rows each.
_ROWS = [
    ("Alice Baker", 1, -0.8, 0.9, -0.5, 0),
    ("Bob Cook", 1, 0.7, -0.4, 0.8, 1),
    ("Carol Diaz", 2, -0.6, 0.5, -0.2, 2),
    ("Dan Evans", 1, 0.9, 0.3, -0.7, 3),
    ("Erin Fox", 1, -0.5, -0.3, 0.6, 4),
    ("Frank Gray", 3, 0.4, 0.7, -0.1, 3),
    ("Gina Hill", 1, -0.9, 0.2, -0.6, 1),
    ("Hank Ives", 1, 0.6, -0.2, 0.4, 4),
]

# _lfw_build_images_npz crops rows 70:195 and cols 78:172 from each source
# image, then halves the crop: the flattened feature length is fixed by those
# slices, and source images must be at least 195x172.
_EXPECTED_FEATURES = 62 * 47 * 3


def _fake_attributes_text() -> str:
    header = "\t".join(["person", "imagenum", "Male", "White", "Black", *_AGE_COLS])
    lines = ["# LFW Attribute descriptions", f"#\t{header}"]
    for person, num, male, white, black, age_idx in _ROWS:
        ages = [0.8 if j == age_idx else -0.9 for j in range(len(_AGE_COLS))]
        values = [str(v) for v in (male, white, black, *ages)]
        lines.append("\t".join([person, str(num), *values]))
    return "\n".join(lines) + "\n"


@pytest.fixture
def mock_lfw_downloads(mocker, make_jpeg_bytes):
    """Patch both network boundaries; return (gdown_mock, fetch_mock)."""

    def _plant_lfw_images(data_home: Path) -> None:
        rng = np.random.default_rng(0)
        for person, num, *_ in _ROWS:
            name = person.replace(" ", "_")
            img_dir = data_home / "lfw_home" / "lfw_funneled" / name
            img_dir.mkdir(parents=True, exist_ok=True)
            _ = (img_dir / f"{name}_{str(num).zfill(4)}.jpg").write_bytes(
                make_jpeg_bytes(rng, 200, 200)
            )

    def _download(id: str, output: str, quiet: bool = False) -> None:
        _ = Path(output).write_text(_fake_attributes_text(), encoding="utf-8")

    def _fetch(color: bool = True, data_home: Path | str | None = None):
        assert data_home is not None
        _plant_lfw_images(Path(data_home))

    gdown_mock = mocker.patch("gdown.download", side_effect=_download)
    fetch_mock = mocker.patch(
        "amulet.datasets.__tabular_datasets.fetch_lfw_people", side_effect=_fetch
    )
    return gdown_mock, fetch_mock


def test_cold_start_downloads_raw_and_builds_caches(
    tmp_path: Path, mock_lfw_downloads
) -> None:
    gdown_mock, fetch_mock = mock_lfw_downloads

    load_lfw(path=tmp_path, target="age", attribute_1="race", attribute_2="gender")

    assert gdown_mock.call_count == 1
    assert fetch_mock.call_count == 1
    assert (tmp_path / "lfw_attributes.txt").exists()
    assert (tmp_path / "lfw_images.npz").exists()
    assert (
        tmp_path / "lfw_processed__target=age__attr1=race__attr2=gender.npz"
    ).exists()


def test_cold_start_returns_well_formed_bundle(
    tmp_path: Path, mock_lfw_downloads
) -> None:
    data = load_lfw(
        path=tmp_path, target="age", attribute_1="race", attribute_2="gender"
    )

    assert data.x_train is not None and data.x_test is not None
    assert data.y_train is not None and data.y_test is not None
    assert data.z_train is not None
    assert len(data.y_train) + len(data.y_test) == len(_ROWS)
    assert data.x_train.dtype == np.float32
    assert data.x_train.shape[1] == _EXPECTED_FEATURES
    assert data.num_features == _EXPECTED_FEATURES
    assert data.y_train.dtype == np.int64
    assert set(np.concatenate([data.y_train, data.y_test])) <= {0, 1}
    assert data.z_train.shape[1] == 2
    assert data.num_classes == 2


def test_cache_hit_skips_download_and_reproduces_split(
    tmp_path: Path, mock_lfw_downloads
) -> None:
    gdown_mock, fetch_mock = mock_lfw_downloads
    first = load_lfw(
        path=tmp_path, target="age", attribute_1="race", attribute_2="gender"
    )
    gdown_mock.reset_mock()
    fetch_mock.reset_mock()

    second = load_lfw(
        path=tmp_path, target="age", attribute_1="race", attribute_2="gender"
    )

    gdown_mock.assert_not_called()
    fetch_mock.assert_not_called()
    assert first.y_train is not None and second.y_train is not None
    np.testing.assert_array_equal(first.y_train, second.y_train)


def test_changed_params_reprocess_without_redownload(
    tmp_path: Path, mock_lfw_downloads
) -> None:
    gdown_mock, fetch_mock = mock_lfw_downloads
    load_lfw(path=tmp_path, target="age", attribute_1="race", attribute_2="gender")
    gdown_mock.reset_mock()
    fetch_mock.reset_mock()

    data = load_lfw(
        path=tmp_path, target="gender", attribute_1="race", attribute_2="age"
    )

    gdown_mock.assert_not_called()
    fetch_mock.assert_not_called()
    assert (
        tmp_path / "lfw_processed__target=gender__attr1=race__attr2=age.npz"
    ).exists()
    assert (
        tmp_path / "lfw_processed__target=age__attr1=race__attr2=gender.npz"
    ).exists()
    assert data.y_train is not None and data.y_test is not None
    assert set(np.concatenate([data.y_train, data.y_test])) <= {0, 1}


@pytest.fixture
def make_attributes_txt():
    """Minimal two-row attributes file text in raw or pre-cleaned format."""

    def _make(*, raw_format: bool) -> str:
        header = "person\timagenum\tMale\tSmiling"
        rows = "Alice\t1\t0.8\t-0.3\nBob\t2\t-0.5\t0.4\n"
        if raw_format:
            return f"# LFW Attribute descriptions\n#\t{header}\n{rows}"
        return f"{header}\n{rows}"

    return _make


@pytest.mark.parametrize("raw_format", [True, False])
def test_read_attributes_parses_both_header_formats(
    tmp_path: Path, make_attributes_txt, raw_format: bool
) -> None:
    attrs_file = tmp_path / "lfw_attributes.txt"
    _ = attrs_file.write_text(
        make_attributes_txt(raw_format=raw_format), encoding="utf-8"
    )

    df = _lfw_read_attributes(attrs_file)

    assert list(df.columns) == ["person", "imagenum", "Male", "Smiling"]
    assert len(df) == 2


def test_read_attributes_formats_yield_identical_dataframes(
    tmp_path: Path, make_attributes_txt
) -> None:
    raw_file = tmp_path / "raw.txt"
    clean_file = tmp_path / "clean.txt"
    _ = raw_file.write_text(make_attributes_txt(raw_format=True), encoding="utf-8")
    _ = clean_file.write_text(make_attributes_txt(raw_format=False), encoding="utf-8")

    pd.testing.assert_frame_equal(
        _lfw_read_attributes(raw_file), _lfw_read_attributes(clean_file)
    )


def test_attr_labels_gender_maps_sign_of_male_column() -> None:
    attributes = pd.DataFrame({"Male": [1.0, -1.0, 0.5, -0.5]})

    labels = _lfw_attr_labels(attributes, "gender")

    assert labels == {0: 1, 1: 0, 2: 1, 3: 0}


def test_attr_labels_age_argmax_and_threshold_drop() -> None:
    rows = [
        [0.9, -0.9, -0.9, -0.9, -0.9],  # argmax 0
        [-0.9, -0.9, -0.9, -0.9, 0.8],  # argmax 4
        [-0.9, -0.9, -0.9, -0.9, -0.9],  # every score below -0.1: row dropped
    ]
    attributes = pd.DataFrame(rows, columns=_AGE_COLS)  # type: ignore[reportArgumentType]

    labels = _lfw_attr_labels(attributes, "age")

    assert labels == {0: 0, 1: 4}


@pytest.mark.parametrize("attribute", ["height", "smile", ""])
def test_attr_labels_unsupported_attribute_raises(attribute: str) -> None:
    attributes = pd.DataFrame({"Male": [1.0]})

    with pytest.raises(ValueError):
        _ = _lfw_attr_labels(attributes, attribute)
