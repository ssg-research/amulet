"""Contract for the E3 table renderer (plan S8 Level 1, S7.1 table contract).

A pure function of the committed CSV: hand-checked tiny CSV in, exact `.tex`
fragment out. Numbers are chosen so each mean and standard error is checkable by
eye. The baseline measures attribute inference against the clean model and has
no defended model to perturb, so its two robust-accuracy columns are always
blank; the budget rows measure it against $\\modeldef$ and fill every column.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

from make.make_tab_attinf_advrtr import coverage, render_table

from experiments.e3_advtr_attrinf.schemas import SCHEMA

if TYPE_CHECKING:
    from pathlib import Path

_FILLER: dict[str, object] = {
    "arch": "linearnet",
    "capacity": "m1",
    "training_size": 1.0,
    "epochs": 100,
    "batch_size": 128,
    "adv_train_fraction": 0.5,
    "step_size": 0.0025,
    "iterations": 40,
    "sensitive_attr_1": "race",
    "sensitive_attr_2": "sex",
    "timestamp": "2026-07-21T00:00:00",
}


def _write(directory: Path, rows: list[dict[str, object]]) -> Path:
    """Write an E3 result CSV carrying the full schema header."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "e3_advtr_attrinf.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEMA.header)
        writer.writeheader()
        writer.writerows([{**_FILLER, **row} for row in rows])
    return path


def _baseline(
    seed: int,
    dataset: str,
    *,
    test_acc: float,
    acc_att_race: float,
    auc_race: float,
    acc_att_sex: float,
    auc_sex: float,
) -> dict[str, object]:
    """One baseline row: attribute inference against the clean $\\modelstd$."""
    return {
        "exp_id": seed,
        "dataset": dataset,
        "epsilon": 0.0,
        "model_role": "baseline",
        "test_acc": test_acc,
        "target_robust_acc": "",
        "defended_robust_acc": "",
        "acc_att_race": acc_att_race,
        "auc_race": auc_race,
        "acc_att_sex": acc_att_sex,
        "auc_sex": auc_sex,
    }


def _defended(
    seed: int,
    dataset: str,
    epsilon: float,
    *,
    test_acc: float,
    target_robust_acc: float,
    defended_robust_acc: float,
    acc_att_race: float,
    auc_race: float,
    acc_att_sex: float,
    auc_sex: float,
) -> dict[str, object]:
    """One defended row: attribute inference against $\\modeldef$ at a budget."""
    return {
        "exp_id": seed,
        "dataset": dataset,
        "epsilon": epsilon,
        "model_role": "defended",
        "test_acc": test_acc,
        "target_robust_acc": target_robust_acc,
        "defended_robust_acc": defended_robust_acc,
        "acc_att_race": acc_att_race,
        "auc_race": auc_race,
        "acc_att_sex": acc_att_sex,
        "auc_sex": auc_sex,
    }


def _census_two_seeds(directory: Path) -> Path:
    """census with two seeds: a baseline row and one budget row per seed."""
    rows = [
        _baseline(
            0,
            "census",
            test_acc=81.0,
            acc_att_race=58.0,
            auc_race=0.60,
            acc_att_sex=65.0,
            auc_sex=0.68,
        ),
        _baseline(
            1,
            "census",
            test_acc=83.0,
            acc_att_race=60.0,
            auc_race=0.62,
            acc_att_sex=67.0,
            auc_sex=0.70,
        ),
        _defended(
            0,
            "census",
            0.01,
            test_acc=82.0,
            target_robust_acc=73.0,
            defended_robust_acc=81.0,
            acc_att_race=59.0,
            auc_race=0.62,
            acc_att_sex=65.0,
            auc_sex=0.70,
        ),
        _defended(
            1,
            "census",
            0.01,
            test_acc=82.0,
            target_robust_acc=75.0,
            defended_robust_acc=81.0,
            acc_att_race=59.0,
            auc_race=0.62,
            acc_att_sex=67.0,
            auc_sex=0.70,
        ),
    ]
    return _write(directory, rows)


def test_the_baseline_row_leaves_the_robust_columns_blank(tmp_path: Path) -> None:
    """Attribute inference against the clean model: the two robust columns are dashes.

    There is no defended model at baseline to perturb, so those cells stay blank
    however many seeds ran. The other cells pool per seed: test accuracy
    {81, 83} -> 82.00 $\\pm$ 1.00, race AUC {0.60, 0.62} -> 0.61 $\\pm$ 0.01.
    """
    rendered = render_table(_census_two_seeds(tmp_path))

    assert (
        "        Baseline ($\\modelstd$) & 82.00~$\\pm$~1.00 & - & - "
        "& 59.00~$\\pm$~1.00 & 0.61~$\\pm$~0.01 & 66.00~$\\pm$~1.00 & 0.69~$\\pm$~0.01 \\\\"
        in rendered.splitlines()
    )


def test_a_budget_row_fills_both_robust_columns(tmp_path: Path) -> None:
    """A defended row carries both robust accuracies; robust-std {73, 75} -> 74.00 $\\pm$ 1.00."""
    rendered = render_table(_census_two_seeds(tmp_path))

    assert (
        "        $\\epsilon_{rob}$ = 0.01 & 82.00~$\\pm$~0.00 & 74.00~$\\pm$~1.00 "
        "& 81.00~$\\pm$~0.00 & 59.00~$\\pm$~0.00 & 0.62~$\\pm$~0.00 & 66.00~$\\pm$~1.00 & 0.70~$\\pm$~0.00 \\\\"
        in rendered.splitlines()
    )


def test_a_single_seed_baseline_has_no_error_term(tmp_path: Path) -> None:
    """One seed has no spread, so each attribute cell is the bare value."""
    path = _write(
        tmp_path,
        [
            _baseline(
                0,
                "lfw",
                test_acc=82.84,
                acc_att_race=66.80,
                auc_race=0.73,
                acc_att_sex=76.59,
                auc_sex=0.84,
            )
        ],
    )

    rendered = render_table(path)

    assert (
        "        Baseline ($\\modelstd$) & 82.84 & - & - & 66.80 & 0.73 & 76.59 & 0.84 \\\\"
        in rendered.splitlines()
    )


def test_both_dataset_blocks_are_present_in_order(tmp_path: Path) -> None:
    """The table carries a census block then an lfw block."""
    rendered = render_table(_census_two_seeds(tmp_path))

    assert "\\multicolumn{8}{c}{\\textbf{\\census}}" in rendered
    assert "\\multicolumn{8}{c}{\\textbf{\\lfw}}" in rendered
    assert rendered.index("\\textbf{\\census}") < rendered.index("\\textbf{\\lfw}")


def test_rendering_an_absent_csv_produces_a_full_blank_table(tmp_path: Path) -> None:
    """With nothing measured the structure still renders, every cell blank."""
    rendered = render_table(tmp_path / "does_not_exist.csv")

    assert rendered.startswith("\\begin{table*}")
    assert rendered.endswith("\\end{table*}\n")
    assert "$\\pm$" not in rendered


def test_coverage_names_the_budgets_with_no_data(tmp_path: Path) -> None:
    """Coverage reports the baseline and budget seed counts, MISSING where none."""
    lines = coverage(_census_two_seeds(tmp_path))

    assert any(
        "census" in line and "baseline" in line and "2 seed(s)" in line
        for line in lines
    )
    assert any(
        "census" in line and "eps=0.06" in line and "MISSING" in line for line in lines
    )
    assert any(
        "lfw" in line and "baseline" in line and "MISSING" in line for line in lines
    )


def test_the_rendered_table_carries_a_block_of_rows_per_dataset(
    tmp_path: Path,
) -> None:
    """Every dataset gets its heading, its baseline, and one row per budget.

    The paper's table is not mirrored in this repository, so the shape is
    asserted against the renderer's own declared layout: per dataset, a
    `multicolumn` heading, a column-header line, then the baseline row plus one
    row per swept budget.
    """
    from experiments.e3_advtr_attrinf.schemas import DATASETS, EPSILONS

    rendered = render_table(_census_two_seeds(tmp_path))

    rows_per_dataset = 2 + 1 + len(EPSILONS)  # 2 heading lines, baseline, budgets
    assert rendered.count("\\\\\n") == len(DATASETS) * rows_per_dataset
    for dataset in DATASETS:
        assert f"\\{dataset}" in rendered
