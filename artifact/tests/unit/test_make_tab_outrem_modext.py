"""Contract for the E4 table renderer (plan S8 Level 1, S7.1 table contract).

Rendering is a pure function of the committed CSV, so every case is a
hand-checked tiny CSV in, an exact `.tex` fragment out: no GPU, no model, no
download. Numbers are chosen so each mean and standard error is checkable by
eye. A two-seed cell of {94, 96} has mean 95.00 and standard error 1.00.

E4 is a reconstruction: no original CSV survived, so the numeric-reproduction
test against the reference `.tex` is deferred until an L3 run produces data
(plan S13.3). These tests pin the renderer's *structure* against the reference,
never its values, so a completed sweep drops straight in.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

from make.make_tab_outrem_modext import coverage, render_table

from experiments.e4_outrem_modext.schemas import SCHEMA

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
    "timestamp": "2026-07-21T00:00:00",
}


def _write(directory: Path, rows: list[dict[str, object]]) -> Path:
    """Write an E4 result CSV carrying the full schema header."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "e4_outrem_modext.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEMA.header)
        writer.writeheader()
        writer.writerows([{**_FILLER, **row} for row in rows])
    return path


def _cell(
    seed: int,
    dataset: str,
    percent: int,
    *,
    defended_test_acc: float,
    stolen_test_acc: float,
    fidelity: float,
    correct_fidelity: float,
) -> dict[str, object]:
    """One E4 result row: the cell it identifies plus the metrics it measured."""
    return {
        "exp_id": seed,
        "dataset": dataset,
        "percent": percent,
        "defended_test_acc": defended_test_acc,
        "stolen_test_acc": stolen_test_acc,
        "fidelity": fidelity,
        "correct_fidelity": correct_fidelity,
    }


def _census_two_seeds(directory: Path) -> Path:
    """census with two seeds at the baseline (0%) and 10% removal.

    Per-percent bases so the two columns differ; a +0/+2 per-seed offset so every
    cell is `mean $\\pm$ 1.00`. Baseline (0%): defended acc base 84, fidelity 94.
    10% removal: defended acc base 81, fidelity 90.
    """
    rows: list[dict[str, object]] = []
    for seed, offset in [(0, 0.0), (1, 2.0)]:
        for percent, acc_base, fid_base in [(0, 84.0, 94.0), (10, 81.0, 90.0)]:
            rows.append(
                _cell(
                    seed,
                    "census",
                    percent,
                    defended_test_acc=acc_base + offset,
                    stolen_test_acc=acc_base + offset,
                    fidelity=fid_base + offset,
                    correct_fidelity=acc_base + offset,
                )
            )
    return _write(directory, rows)


def test_the_baseline_column_is_a_real_measurement(tmp_path: Path) -> None:
    """The $\\modelstd$ column (removal 0) carries real numbers, not blanks.

    Unlike E2's baseline row, E4's zero-removal column is a full model-extraction
    of the clean model: every one of the four metric rows has a value in it.
    defended acc {84, 86} -> 85.00 $\\pm$ 1.00 at 0%, {81, 83} -> 82.00 $\\pm$ 1.00
    at 10%; fidelity {94, 96} -> 95.00 $\\pm$ 1.00 at 0%, {90, 92} -> 91.00 at 10%.
    """
    rendered = render_table(_census_two_seeds(tmp_path)).splitlines()

    assert (
        "    $Acc_{te}$ & 85.00~$\\pm$~1.00 & 82.00~$\\pm$~1.00 & - & - & - \\\\"
        in rendered
    )
    assert (
        "    $\\modelstol$: $Fid$ & 95.00~$\\pm$~1.00 & 91.00~$\\pm$~1.00 & - & - & - \\\\"
        in rendered
    )


def test_a_single_seed_cell_has_no_error_term(tmp_path: Path) -> None:
    """One seed has no spread, so a removal cell is the bare value (plan S1)."""
    path = _write(
        tmp_path,
        [
            _cell(
                0,
                "cifar",
                10,
                defended_test_acc=75.35,
                stolen_test_acc=75.81,
                fidelity=84.66,
                correct_fidelity=70.17,
            )
        ],
    )

    rendered = render_table(path).splitlines()

    assert "    $Acc_{te}$ & - & 75.35 & - & - & - \\\\" in rendered
    assert "    $\\modelstol$: $Fid_{cor}$ & - & 70.17 & - & - & - \\\\" in rendered


def test_a_dataset_with_no_data_renders_every_cell_blank(tmp_path: Path) -> None:
    """A dataset nothing has been run for still renders its block, all dashes."""
    rendered = render_table(_census_two_seeds(tmp_path))

    assert "\\multicolumn{6}{c}{\\textbf{\\fmnist}}" in rendered
    assert "    $Acc_{te}$ & - & - & - & - & - \\\\" in rendered.splitlines()


def test_all_four_dataset_blocks_are_present_in_order(tmp_path: Path) -> None:
    """The table carries census, lfw, fmnist and cifar blocks, in that order."""
    rendered = render_table(_census_two_seeds(tmp_path))
    positions = [
        rendered.index(f"\\textbf{{\\{name}}}")
        for name in ("census", "lfw", "fmnist", "cifar")
    ]

    assert positions == sorted(positions)


def test_rendering_an_absent_csv_produces_a_full_blank_table(tmp_path: Path) -> None:
    """With nothing measured the structure still renders, every cell blank."""
    rendered = render_table(tmp_path / "does_not_exist.csv")

    assert rendered.startswith("\\begin{table*}")
    assert rendered.endswith("\\end{table*}\n")
    assert "$\\pm$" not in rendered


def test_coverage_names_the_removal_levels_with_no_data(tmp_path: Path) -> None:
    """Coverage reports seed counts per cell and says MISSING where there are none."""
    lines = coverage(_census_two_seeds(tmp_path))

    assert any(
        "census" in line and "M_std" in line and "2 seed(s)" in line for line in lines
    )
    assert any(
        "census" in line and "40%" in line and "MISSING" in line for line in lines
    )
    assert any(
        "fmnist" in line and "M_std" in line and "MISSING" in line for line in lines
    )


def test_the_reference_table_and_a_rendered_one_have_the_same_shape(
    tmp_path: Path,
) -> None:
    """The generated table carries a row and rule wherever the paper's does.

    E4 is a reconstruction with no source CSV, so this pins structure, not
    values: the same count of data rows and rules, so a completed L3 sweep drops
    straight in (plan S7.1, S13.3). Numeric reproduction is deferred until then.
    """
    from common.paths import artifact_root

    reference = (artifact_root() / "tables" / "tab_outrem_modext.tex").read_text()
    rendered = render_table(_census_two_seeds(tmp_path))

    assert rendered.count("\\\\\n") == reference.count("\\\\\n")
    assert rendered.count("\\midrule") == reference.count("\\midrule")
