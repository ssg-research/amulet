"""Render `tab_attinf_advrtr` (E3) from the committed result CSV.

    python artifact/make/make_tab_attinf_advrtr.py

Reads `artifact/results/e3_advtr_attrinf.csv` and writes
`artifact/tables/generated/tab_attinf_advrtr.tex`. A pure function of the CSV: no
GPU, no model, no download (plan S13, decision 2).

**Multi-dataset blocks.** One block per dataset (census, lfw), each a
`Baseline ($\\modelstd$)` row measuring attribute inference against the clean
model, plus one row per budget measuring it against the adversarially-trained
$\\modeldef$. Both sensitive attributes (race and sex) get an accuracy and an AUC
column.

**Seed-count agnostic.** Cells aggregate over whatever seeds the CSV holds. The
baseline's robust-accuracy columns are structurally blank: there is no defended
model to perturb.

**Output goes to `tables/generated/`, never over the reference** (plan S7.1).
Numbers must match; formatting is free.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from make.tables_common import BLANK, cell, seed_count

from common.io import read_rows, results_path
from common.paths import artifact_root
from experiments.e3_advtr_attrinf.schemas import (
    DATASETS,
    EPSILONS,
    EXPERIMENT_ID,
    SCHEMA,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

TABLE_STEM = "tab_attinf_advrtr"

_INDENT = " " * 8

_DATASET_MACRO: dict[str, str] = {"census": "\\census", "lfw": "\\lfw"}

# The seven data columns after the row label, in table order.
_COLUMN_LABELS: tuple[str, ...] = (
    "$Acc_{te}$",
    "$Acc_{rob}$ ($\\modelstd$)",
    "$Acc_{rob}$ ($\\modeldef$)",
    "$Acc_{att}$ (\\race)",
    "$AUC$ (\\race)",
    "$Acc_{att}$ (\\sex)",
    "$AUC$ (\\sex)",
)

# The attribute-inference columns, shared by the baseline and budget rows.
_ATTACK_COLUMNS: tuple[str, ...] = (
    "acc_att_race",
    "auc_race",
    "acc_att_sex",
    "auc_sex",
)

_PREAMBLE = """\
\\begin{table*}[!htb]
    \\setlength\\tabcolsep{2.5pt}
    \\footnotesize
    \\centering
    \\caption{Interaction between~\\ref{advtr} Adversarial Training and~\\ref{attinf} Attribute Inference on \\census and \\lfw. Adversarial training does not consistently change susceptibility to~\\ref{attinf}.}
    \\label{tab:attinf_advtr}
    \\begin{tabular}{ c c c c c c c c }
        \\toprule
"""

_EPILOGUE = """\
        \\bottomrule
    \\end{tabular}
\\end{table*}
"""


def _row(label: str, cells: Sequence[str]) -> str:
    """Lay out one table row: a label column followed by its cells."""
    return f"{_INDENT}{label} & {' & '.join(cells)} \\\\\n"


def _rows_for_dataset(
    rows: Sequence[Mapping[str, str]], dataset: str
) -> list[Mapping[str, str]]:
    """Return only the rows belonging to one dataset."""
    return [row for row in rows if row["dataset"] == dataset]


def _baseline_rows(
    rows: Sequence[Mapping[str, str]],
) -> list[Mapping[str, str]]:
    """Return the clean-baseline rows (attribute inference against $\\modelstd$)."""
    return [row for row in rows if row["model_role"] == "baseline"]


def _epsilon_rows(
    rows: Sequence[Mapping[str, str]], epsilon: float
) -> list[Mapping[str, str]]:
    """Return the defended rows for one budget."""
    matched: list[Mapping[str, str]] = []
    for row in rows:
        if row["model_role"] != "defended":
            continue
        try:
            if float(row["epsilon"]) == epsilon:
                matched.append(row)
        except ValueError:
            continue
    return matched


def _block(dataset: str, rows: Sequence[Mapping[str, str]]) -> str:
    """Render one dataset's header, column labels, baseline and budget rows."""
    dataset_rows = _rows_for_dataset(rows, dataset)

    body = f"{_INDENT}\\multicolumn{{8}}{{c}}{{\\textbf{{{_DATASET_MACRO[dataset]}}}}}\\\\\n"
    body += f"{_INDENT}    & {' & '.join(_COLUMN_LABELS)} \\\\\n"
    body += f"{_INDENT}\\midrule\n"

    # Baseline: test accuracy and both attributes measured on the clean model;
    # no defended model exists, so the two robust-accuracy columns are blank.
    baseline = _baseline_rows(dataset_rows)
    baseline_cells = [
        cell(baseline, "test_acc"),
        BLANK,
        BLANK,
        *(cell(baseline, column) for column in _ATTACK_COLUMNS),
    ]
    body += _row("Baseline ($\\modelstd$)", baseline_cells)

    for epsilon in EPSILONS:
        eps_rows = _epsilon_rows(dataset_rows, epsilon)
        cells = [
            cell(eps_rows, "test_acc"),
            cell(eps_rows, "target_robust_acc"),
            cell(eps_rows, "defended_robust_acc"),
            *(cell(eps_rows, column) for column in _ATTACK_COLUMNS),
        ]
        body += _row(f"$\\epsilon_{{rob}}$ = {epsilon:g}", cells)

    return body


def render_table(results_file: Path) -> str:
    """Render the whole table from the E3 result CSV.

    Args:
        results_file: Path to `e3_advtr_attrinf.csv`. May be absent.

    Returns:
        The `.tex` source, trailing-whitespace-free with a single final newline.

    Raises:
        ValueError: If the CSV's header does not match the schema.
    """
    rows = read_rows(results_file)
    if rows and tuple(rows[0]) != SCHEMA.header:
        raise ValueError(
            f"{results_file} does not carry the expected header. "
            f"Found: {', '.join(rows[0])}. Expected: {', '.join(SCHEMA.header)}."
        )

    blocks = [_block(dataset, rows) for dataset in DATASETS]
    return _PREAMBLE + f"{_INDENT}\\midrule\n".join(blocks) + _EPILOGUE


def coverage(results_file: Path) -> list[str]:
    """List each cell's seed count, naming the budgets with no data.

    Args:
        results_file: Path to `e3_advtr_attrinf.csv`.

    Returns:
        One human-readable line per (dataset, budget) cell.
    """
    rows = read_rows(results_file)
    lines: list[str] = []
    for dataset in DATASETS:
        dataset_rows = _rows_for_dataset(rows, dataset)
        baseline_seeds = seed_count(_baseline_rows(dataset_rows))
        status = (
            f"{len(baseline_seeds)} seed(s) {baseline_seeds}"
            if baseline_seeds
            else "MISSING"
        )
        lines.append(f"  {dataset:>8} baseline: {status}")
        for epsilon in EPSILONS:
            seeds = seed_count(_epsilon_rows(dataset_rows, epsilon))
            status = f"{len(seeds)} seed(s) {seeds}" if seeds else "MISSING"
            lines.append(f"  {dataset:>8} eps={epsilon:g}: {status}")
    return lines


def generate(
    results_dir: Path | None = None, out_dir: Path | None = None
) -> list[Path]:
    """Render the E3 table from a results base dir into a generated-output dir.

    Args:
        results_dir: Base directory holding the result CSV, in the shared
            layout. None reads the committed `results/`; a `runs/<level>/`
            directory renders a reviewer's re-run instead.
        out_dir: Directory the `.tex` is written to. None uses
            `tables/generated/`.

    Returns:
        The paths written (one `.tex`).
    """
    results_file = results_path(EXPERIMENT_ID, base=results_dir)
    out_dir = artifact_root() / "tables" / "generated" if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{TABLE_STEM}.tex"
    _ = output.write_text(render_table(results_file))
    return [output]


def coverage_report(results_dir: Path | None = None) -> list[str]:
    """Return per-cell coverage lines for the E3 table from a results base dir."""
    return coverage(results_path(EXPERIMENT_ID, base=results_dir))


def main() -> None:
    """Render the table from a results directory and report cell coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Base results directory to render from. Default: the committed results/.",
    )
    args = parser.parse_args()

    for path in generate(results_dir=args.results_dir):
        print(f"wrote {path}")
    print("cell coverage:")
    for line in coverage_report(results_dir=args.results_dir):
        print(line)


if __name__ == "__main__":
    main()
