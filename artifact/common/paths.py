"""Resolve the artifact's directory roots from this file's own location.

A reviewer must be able to run any script in `artifact/` from any working
directory with nothing to configure: no absolute paths, no `--root` flag, no
`sys.path.append("../../")`. Every path in the artifact is derived from the two
functions here, which walk up from `__file__`, so the answer is the same
whatever the process CWD is and survives the tree being moved or renamed.

Entry scripts that need `common` importable bootstrap themselves with::

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

using the computed path, then `from common.paths import repo_root`.
"""

from pathlib import Path

# artifact/common/paths.py -> parents[0] = common, [1] = artifact, [2] = repo root.
_THIS_FILE = Path(__file__).resolve()


def artifact_root() -> Path:
    """Return the absolute path of the `artifact/` directory.

    Returns:
        The directory holding `common/`, `experiments/`, `make/` and `results/`.
    """
    return _THIS_FILE.parents[1]


def repo_root() -> Path:
    """Return the absolute path of the repository root.

    This is the directory holding the `amulet/` library the artifact exercises,
    and the `data/` cache that `amulet.utils.load_data` reads and writes.

    Returns:
        The repository root directory.
    """
    return _THIS_FILE.parents[2]
