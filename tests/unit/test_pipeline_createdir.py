from pathlib import Path

from amulet.utils.__pipeline import create_dir


def test_create_dir_new(tmp_path):
    target = tmp_path / "new_dir"
    assert not target.exists()

    resolved = create_dir(target)

    assert resolved == target.resolve()
    assert target.exists()
    assert target.is_dir()


def test_create_dir_exists(tmp_path):
    target = tmp_path / "existing_dir"
    target.mkdir()
    assert target.exists()

    resolved = create_dir(target)

    assert resolved == target.resolve()
    assert target.exists()


def test_create_dir_str_path(tmp_path):
    target_str = str(tmp_path / "str_dir")

    resolved = create_dir(target_str)

    assert isinstance(resolved, Path)
    assert resolved.exists()
    assert str(resolved) == str(Path(target_str).resolve())
