"""The results/runs split: no run() default touches the committed ground truth (P5).

`artifact/results/` holds committed, read-only ground-truth CSVs (the shipped
single-seed data a `make_*` renders from). Every experiment's `run()` writes
under a gitignored `artifact/runs/<level>/` tree instead, whatever the level, so
neither a reviewer's `full` re-run nor a `smoke`/`test` run can clobber or dilute
the shipped numbers.

These are the promotion (per legacy-code-rescue) of the characterization that
pinned the old behavior (`full` wrote into `results/`, `smoke` into
`results/<id>/smoke`): the assertion here is the *intent* — every default output
directory is under `runs/<level>/` and none is under `results/` — not the code's
former output. They also pin the layout parity that lets one renderer read either
tree: `runs/<level>/` mirrors `results/` exactly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from common.config import LevelConfig, get_level
from common.io import results_path, results_root, run_output_dir, runs_root
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
def test_no_default_output_dir_is_under_results(name: str) -> None:
    """At every level, no experiment's default output directory is under results/.

    This is the load-bearing isolation property: the committed ground truth in
    `results/` can never be written by a run, so a reviewer's re-run (or a
    forgotten `full`) leaves the shipped CSVs byte-for-byte intact.
    """
    committed = results_root()
    for label, directory in _all_default_dirs(name).items():
        assert committed != directory and committed not in directory.parents, (
            f"{label} at level {name} writes under results/: {directory}"
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
def test_runs_tree_mirrors_the_results_layout(name: str) -> None:
    """`runs/<level>/` uses the same per-experiment layout as `results/`.

    E1 and E5 write into a `<experiment_id>/` subdirectory (they emit several
    CSVs); E2/E3/E4 write a single `<experiment_id>.csv` directly into the level
    root. This parity is what lets a `make_*` renderer read a reviewer's
    `runs/full/` with the exact path logic it uses for `results/`.
    """
    dirs = _all_default_dirs(name)
    level_root = run_output_dir(name)

    assert dirs["e1"] == level_root / "e1_attack_baselines"
    assert dirs["e5_onion"] == level_root / "e5_textbadnets"
    assert dirs["e5_dp"] == level_root / "e5_textbadnets"
    for single in ("e2", "e3", "e4"):
        assert dirs[single] == level_root


def test_results_path_base_swaps_the_root_but_keeps_the_layout() -> None:
    """`results_path(base=runs/full)` resolves the same relative layout as under results/.

    A `make_*` reads shipped data at `results_path(id)` and a reviewer's re-run
    at `results_path(id, base=runs/full)`; both must land on the same filename in
    their respective roots.
    """
    runs_full = run_output_dir("full")

    assert results_path("e2_advtr_modext") == results_root() / "e2_advtr_modext.csv"
    assert (
        results_path("e2_advtr_modext", base=runs_full)
        == runs_full / "e2_advtr_modext.csv"
    )
    assert (
        results_path("e5_textbadnets", "onion", base=runs_full)
        == runs_full / "e5_textbadnets" / "onion.csv"
    )


def test_runs_root_is_the_artifact_runs_directory() -> None:
    """`runs_root()` is `artifact/runs`, the gitignored sibling of `results/`."""
    assert runs_root().name == "runs"
    assert runs_root().parent == results_root().parent


def test_the_registry_still_lists_the_five_experiments() -> None:
    """A guard that the isolation checks cover every registered experiment."""
    assert set(EXPERIMENT_IDS) == {
        "e1_attack_baselines",
        "e2_advtr_modext",
        "e3_advtr_attrinf",
        "e4_outrem_modext",
        "e5_textbadnets",
    }
