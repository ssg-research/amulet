from pathlib import Path

from amulet.utils.__pipeline import create_dir


def test_create_dir_new(tmp_path):
    # Arrange
    target = tmp_path / "new_dir"
    assert not target.exists()

    # Act
    resolved = create_dir(target)

    # Assert
    assert resolved == target.resolve()
    assert target.exists()
    assert target.is_dir()


def test_create_dir_exists(tmp_path):
    # Arrange
    target = tmp_path / "existing_dir"
    target.mkdir()
    assert target.exists()

    # Act
    resolved = create_dir(target)

    # Assert
    assert resolved == target.resolve()
    assert target.exists()


def test_create_dir_str_path(tmp_path):
    # Arrange
    target_str = str(tmp_path / "str_dir")

    # Act
    resolved = create_dir(target_str)

    # Assert
    assert isinstance(resolved, Path)
    assert resolved.exists()
    assert str(resolved) == str(Path(target_str).resolve())
