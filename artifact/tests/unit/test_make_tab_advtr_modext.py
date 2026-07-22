"""Contract for the E2 table renderer (plan S8 Level 1, S7.1 table contract).

Rendering is a pure function of the result CSV, so every case is a
hand-checked tiny CSV in, an exact `.tex` fragment out: no GPU, no model, no
download. Numbers are chosen so each mean and standard error is checkable by
eye. A two-seed cell of {94, 96} has mean 95.00 and standard error 1.00; the
baseline pools one clean test accuracy per seed, {82, 84}, to 83.00 $\\pm$ 1.00.

Two kinds of blank are covered: the baseline's five columns are *structurally*
blank (there is no defended or stolen model to measure), and a budget nothing
has been run for is blank too but reported by `coverage` as MISSING.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

from make.make_tab_advtr_modext import coverage, render_table
from make.tables_common import format_cell, mean_and_standard_error

from experiments.e2_advtr_modext.schemas import SCHEMA

if TYPE_CHECKING:
    from pathlib import Path

# Columns the renderer never reads, filled with one constant so the fixtures
# carry the real header without burying the numbers under test.
_FILLER: dict[str, object] = {
    "arch": "linearnet",
    "capacity": "m1",
    "training_size": 1.0,
    "epochs": 100,
    "batch_size": 256,
    "adv_train_fraction": 0.5,
    "step_size": 0.0025,
    "iterations": 40,
    "timestamp": "2026-07-21T00:00:00",
}


def _write(directory: Path, rows: list[dict[str, object]]) -> Path:
    """Write an E2 result CSV carrying the full schema header."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "e2_advtr_modext.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEMA.header)
        writer.writeheader()
        writer.writerows([{**_FILLER, **row} for row in rows])
    return path


def _cell(
    seed: int,
    dataset: str,
    epsilon: float,
    *,
    target_test_acc: float,
    defended_test_acc: float,
    target_robust_acc: float,
    defended_robust_acc: float,
    stolen_test_acc: float,
    fidelity: float,
    correct_fidelity: float,
) -> dict[str, object]:
    """One E2 result row: the cell it identifies plus the metrics it measured."""
    return {
        "exp_id": seed,
        "dataset": dataset,
        "epsilon": epsilon,
        "target_test_acc": target_test_acc,
        "defended_test_acc": defended_test_acc,
        "target_robust_acc": target_robust_acc,
        "defended_robust_acc": defended_robust_acc,
        "stolen_test_acc": stolen_test_acc,
        "fidelity": fidelity,
        "correct_fidelity": correct_fidelity,
    }


def _census_two_seeds(directory: Path) -> Path:
    """census with two seeds at epsilon 0.01; the clean baseline differs by seed."""
    rows: list[dict[str, object]] = []
    for seed, clean_acc, fid in [(0, 82.0, 94.0), (1, 84.0, 96.0)]:
        rows.append(
            _cell(
                seed,
                "census",
                0.01,
                target_test_acc=clean_acc,
                defended_test_acc=82.0,
                target_robust_acc=73.0,
                defended_robust_acc=81.0,
                stolen_test_acc=83.0,
                fidelity=fid,
                correct_fidelity=84.0,
            )
        )
    return _write(directory, rows)


def test_the_baseline_pools_one_clean_accuracy_per_seed(tmp_path: Path) -> None:
    """The baseline row is `mean $\\pm$ SE` over one clean test accuracy per seed.

    The clean $\\modelstd$ is epsilon-independent, so its accuracy is pooled from
    the dataset's rows (one per seed: 82 and 84 -> 83.00 $\\pm$ 1.00), and the
    five defended/stolen columns are structurally blank on this row.
    """
    rendered = render_table(_census_two_seeds(tmp_path))

    assert (
        "        Baseline ($\\modelstd$) & 83.00~$\\pm$~1.00 & - & - & - & - & - \\\\"
        in rendered.splitlines()
    )


def test_a_budget_row_renders_mean_and_standard_error(tmp_path: Path) -> None:
    """A two-seed budget cell renders as `mean $\\pm$ SE`; fidelity {94, 96} -> 95.00 $\\pm$ 1.00."""
    rendered = render_table(_census_two_seeds(tmp_path))

    assert (
        "        $\\epsilon_{rob}$ = 0.01 & 82.00~$\\pm$~0.00 & 73.00~$\\pm$~0.00 "
        "& 81.00~$\\pm$~0.00 & 83.00~$\\pm$~0.00 & 95.00~$\\pm$~1.00 & 84.00~$\\pm$~0.00 \\\\"
        in rendered.splitlines()
    )


def test_a_single_seed_cell_has_no_error_term(tmp_path: Path) -> None:
    """One seed has no spread, so the budget cell is the bare value (plan S1)."""
    path = _write(
        tmp_path,
        [
            _cell(
                0,
                "cifar",
                0.03,
                target_test_acc=83.3,
                defended_test_acc=70.69,
                target_robust_acc=9.12,
                defended_robust_acc=38.85,
                stolen_test_acc=70.31,
                fidelity=79.17,
                correct_fidelity=79.82,
            )
        ],
    )

    rendered = render_table(path)

    assert (
        "        $\\epsilon_{rob}$ = 0.03 & 70.69 & 9.12 & 38.85 & 70.31 & 79.17 & 79.82 \\\\"
        in rendered.splitlines()
    )


def test_a_dataset_with_no_data_renders_every_cell_blank(tmp_path: Path) -> None:
    """A dataset nothing has been run for still renders its block, all dashes."""
    rendered = render_table(_census_two_seeds(tmp_path))

    # fmnist has no rows in the census-only fixture.
    assert "\\multicolumn{7}{c}{\\textbf{\\fmnist}}" in rendered
    assert (
        "        Baseline ($\\modelstd$) & - & - & - & - & - & - \\\\"
        in rendered.splitlines()
    )


def test_all_four_dataset_blocks_are_present_in_order(tmp_path: Path) -> None:
    """The table carries a census, fmnist, lfw and cifar block, in that order."""
    rendered = render_table(_census_two_seeds(tmp_path))
    positions = [
        rendered.index(f"\\textbf{{\\{name}}}")
        for name in ("census", "fmnist", "lfw", "cifar")
    ]

    assert positions == sorted(positions)


def test_rendering_an_absent_csv_produces_a_full_blank_table(tmp_path: Path) -> None:
    """With nothing measured the structure still renders, every cell blank."""
    rendered = render_table(tmp_path / "does_not_exist.csv")

    assert rendered.startswith("\\begin{table*}")
    assert rendered.endswith("\\end{table*}\n")
    assert "$\\pm$" not in rendered


def test_coverage_names_the_budgets_with_no_data(tmp_path: Path) -> None:
    """Coverage reports seed counts per cell and says MISSING where there are none."""
    lines = coverage(_census_two_seeds(tmp_path))

    assert any(
        "census" in line and "eps=0.01" in line and "2 seed(s)" in line
        for line in lines
    )
    assert any(
        "census" in line and "eps=0.05" in line and "MISSING" in line for line in lines
    )
    assert any(
        "fmnist" in line and "baseline" in line and "MISSING" in line for line in lines
    )


def test_the_rendered_table_carries_a_block_of_rows_per_dataset(
    tmp_path: Path,
) -> None:
    """Every dataset gets its heading, its baseline, and one row per budget.

    Real E2 numbers need many GPU-hours and come from a later step, so this pins
    structure, not values. The paper's table is not mirrored in this repository,
    so the shape is asserted against the renderer's own declared layout.
    """
    from experiments.e2_advtr_modext.schemas import DATASETS, EPSILONS

    rendered = render_table(_census_two_seeds(tmp_path))

    rows_per_dataset = 2 + 1 + len(EPSILONS)  # 2 heading lines, baseline, budgets
    assert rendered.count("\\\\\n") == len(DATASETS) * rows_per_dataset
    for dataset in DATASETS:
        assert f"\\{dataset}" in rendered


def test_mean_and_standard_error_is_the_error_of_the_mean() -> None:
    """{1, 2, 3} has sample stdev 1.0 but standard error 1/sqrt(3), the reported term."""
    import pytest

    assert mean_and_standard_error([1.0, 2.0, 3.0]) == pytest.approx((
        2.0,
        0.5773502691896257,
    ))
    assert format_cell([92.0]) == "92.00"
