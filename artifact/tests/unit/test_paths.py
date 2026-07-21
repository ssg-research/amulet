"""Tests for common/paths.py.

The artifact's path discipline is a hard constraint: a reviewer runs any script
from any working directory with no environment or root variable to set. These
tests pin exactly that — the roots are derived from the module's own location,
so changing the process CWD cannot move them.
"""

import os
from pathlib import Path

import pytest

from common.paths import artifact_root, repo_root


def test_artifact_root_is_the_directory_containing_common() -> None:
    assert (artifact_root() / "common" / "paths.py").is_file()


def test_repo_root_is_the_parent_of_artifact_root() -> None:
    assert repo_root() == artifact_root().parent


def test_repo_root_contains_the_library_and_pyproject() -> None:
    assert (repo_root() / "amulet" / "__init__.py").is_file()
    assert (repo_root() / "pyproject.toml").is_file()


@pytest.mark.parametrize("cwd", ["repo_root", "tmp_path", "home"])
def test_roots_are_independent_of_cwd(cwd: str, tmp_path: Path) -> None:
    expected_artifact, expected_repo = artifact_root(), repo_root()
    targets = {
        "repo_root": expected_repo,
        "tmp_path": tmp_path,
        "home": Path.home(),
    }
    original = Path.cwd()
    try:
        os.chdir(targets[cwd])
        assert artifact_root() == expected_artifact
        assert repo_root() == expected_repo
    finally:
        os.chdir(original)


def test_roots_are_absolute_and_resolved() -> None:
    for root in (artifact_root(), repo_root()):
        assert root.is_absolute()
        assert root == root.resolve()
