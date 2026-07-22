"""Level isolation: a run writes only under its own `runs/<level>/` tree (P5).

No result CSVs ship with the repository, so every number a reviewer sees comes
from a run they performed. Each experiment's `run()` writes under a gitignored
`artifact/runs/<level>/` tree keyed by its level, so a cheap `test` or `smoke`
run can never dilute or overwrite the `full` results a paper comparison rests on.

These tests assert that intent directly: every default output directory is under
`runs/<level>/` for its own level, and no level's tree overlaps another's. They
also pin the layout every `make_*` renderer relies on to read any level's tree
with one path rule.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from common.config import LevelConfig, get_level
from common.io import default_results_dir, results_path, run_output_dir, runs_root
from common.registry import EXPERIMENT_IDS

_LEVELS = ("test", "smoke", "full")


def _level(name: str) -> LevelConfig:
    return get_level(name).with_defaults(epochs=1)


def _all_default_dirs(name: str) -> dict[str, Path]:
    """Every experiment's default output dir at one level, keyed by a label."""
    from experiments import advtr_common as advtr
    from experiments.e1_attack_baselines import shared
    from experiments.e5_textbadnets import dp, onion

    level = _level(name)
    return {
        "e1": shared.default_output_dir(level),
        "e2": advtr.default_output_dir(level, "e2_advtr_modext"),
        "e3": advtr.default_output_dir(level, "e3_advtr_attrinf"),
        "e4": advtr.default_output_dir(level, "e4_outrem_modext"),
        "e5_onion": onion.default_output_dir(level),
        "e5_dp": dp.default_output_dir(level),
    }


@pytest.mark.parametrize("name", _LEVELS)
def test_no_default_output_dir_escapes_its_own_level(name: str) -> None:
    """A run at one level never writes into another level's tree.

    This is the load-bearing isolation property: a cheap `test` or `smoke` run
    cannot land rows in `runs/full/`, where a paper comparison reads from.
    """
    others = [run_output_dir(other) for other in _LEVELS if other != name]
    for label, directory in _all_default_dirs(name).items():
        for other in others:
            assert other != directory and other not in directory.parents, (
                f"{label} at level {name} writes under {other}: {directory}"
            )


@pytest.mark.parametrize("name", _LEVELS)
def test_every_default_output_dir_is_under_runs_for_its_level(name: str) -> None:
    """Every default output directory lives under `runs/<level>/`."""
    level_root = run_output_dir(name)
    for label, directory in _all_default_dirs(name).items():
        assert directory == level_root or level_root in directory.parents, (
            f"{label} at level {name} does not write under {level_root}: {directory}"
        )


@pytest.mark.parametrize("name", _LEVELS)
def test_every_level_tree_uses_the_same_layout(name: str) -> None:
    """Every `runs/<level>/` uses one per-experiment layout.

    E1 and E5 write into a `<experiment_id>/` subdirectory (they emit several
    CSVs); E2/E3/E4 write a single `<experiment_id>.csv` directly into the level
    root. This parity is what lets a `make_*` renderer read any level's tree with
    the same path logic.
    """
    dirs = _all_default_dirs(name)
    level_root = run_output_dir(name)

    assert dirs["e1"] == level_root / "e1_attack_baselines"
    assert dirs["e5_onion"] == level_root / "e5_textbadnets"
    assert dirs["e5_dp"] == level_root / "e5_textbadnets"
    for single in ("e2", "e3", "e4"):
        assert dirs[single] == level_root


def test_results_path_base_swaps_the_root_but_keeps_the_layout() -> None:
    """`results_path(base=...)` resolves one layout under any level root.

    A `make_*` reads a full run at `results_path(id)` and any other level at
    `results_path(id, base=runs/<level>)`; both must land on the same filename in
    their respective roots.
    """
    runs_full = run_output_dir("full")

    assert results_path("e2_advtr_modext") == runs_full / "e2_advtr_modext.csv"
    assert (
        results_path("e2_advtr_modext", base=runs_full)
        == runs_full / "e2_advtr_modext.csv"
    )
    assert (
        results_path("e5_textbadnets", "onion", base=runs_full)
        == runs_full / "e5_textbadnets" / "onion.csv"
    )


def test_runs_root_is_the_artifact_runs_directory() -> None:
    """`runs_root()` is `artifact/runs`, and a full run is its default subtree."""
    from common.paths import artifact_root

    assert runs_root() == artifact_root() / "runs"
    assert default_results_dir() == runs_root() / "full"


def test_the_registry_still_lists_the_five_experiments() -> None:
    """A guard that the isolation checks cover every registered experiment."""
    assert set(EXPERIMENT_IDS) == {
        "e1_attack_baselines",
        "e2_advtr_modext",
        "e3_advtr_attrinf",
        "e4_outrem_modext",
        "e5_textbadnets",
    }
