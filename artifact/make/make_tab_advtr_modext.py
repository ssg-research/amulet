"""Render `tab_advtr_modext` (E2) from the committed result CSV.

    python artifact/make/make_tab_advtr_modext.py

Reads `artifact/results/e2_advtr_modext.csv` and writes
`artifact/tables/generated/tab_advtr_modext.tex`. Rendering is a pure function of
the CSV: no GPU, no model, no download, seconds (plan S13, decision 2).

**Multi-dataset blocks.** One `\\multicolumn` block per dataset (census, fmnist,
lfw, cifar), each a `Baseline ($\\modelstd$)` row plus one row per budget.

**Seed-count agnostic.** A cell aggregates over whatever seeds the CSV holds:
several as `mean ~$\\pm$~ SE`, one as the bare value. The baseline's clean test
accuracy is epsilon-independent, so it is pooled once per seed from every row of
that dataset.

**Output goes to `tables/generated/`.** The paper's own tables are not mirrored
in this repository, so the fixed point a rendered table is compared against is
the paper itself (Table 7).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from make.tables_common import BLANK, cell, group_by, seed_count

from common.io import read_rows, results_path
from common.paths import artifact_root
from experiments.e2_advtr_modext.schemas import (
    DATASETS,
    EPSILONS,
    EXPERIMENT_ID,
    SCHEMA,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

TABLE_STEM = "tab_advtr_modext"

_INDENT = " " * 8

# Dataset name -> the LaTeX macro the reference table uses for it.
_DATASET_MACRO: dict[str, str] = {
    "census": "\\census",
    "fmnist": "\\fmnist",
    "lfw": "\\lfw",
    "cifar": "\\cifar",
}

# The six data columns after the row label, in table order.
_COLUMN_LABELS: tuple[str, ...] = (
    "$Acc_{te}$ ($\\modeldef$)",
    "$Acc_{rob}$ ($\\modelstd$)",
    "$Acc_{rob}$ ($\\modeldef$)",
    "$\\modelstol$: $Acc_{te}$",
    "$\\modelstol$: $Fid$",
    "$\\modelstol$: $Fid_{cor}$",
)

# The CSV columns each epsilon row renders, aligned with `_COLUMN_LABELS`.
_EPS_COLUMNS: tuple[str, ...] = (
    "defended_test_acc",
    "target_robust_acc",
    "defended_robust_acc",
    "stolen_test_acc",
    "fidelity",
    "correct_fidelity",
)

_PREAMBLE = """\
\\begin{table*}[htb]
    \\setlength\\tabcolsep{2.5pt}
    \\centering
    \\footnotesize
    \\caption{Interaction between~\\ref{advtr} Adversarial Training and~\\ref{modelext} Unauthorized Model Ownership. $Fid$ and $Fid_{cor}$ are fidelity and correct fidelity between $\\modeldef$ and the stolen model $\\modelstol$. Adversarial training raises fidelity but lowers correct fidelity on harder datasets.}
    \\label{tab:advtr_modelext}
    \\begin{tabular}{ c c c c c c c }
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


def _block(dataset: str, rows: Sequence[Mapping[str, str]]) -> str:
    """Render one dataset's header, column labels, baseline and budget rows."""
    dataset_rows = _rows_for_dataset(rows, dataset)
    by_epsilon = group_by(dataset_rows, "epsilon")

    body = f"{_INDENT}\\multicolumn{{7}}{{c}}{{\\textbf{{{_DATASET_MACRO[dataset]}}}}}\\\\\n"
    body += f"{_INDENT}    & {' & '.join(_COLUMN_LABELS)} \\\\\n"
    body += f"{_INDENT}\\midrule\n"

    # The clean baseline's test accuracy is epsilon-independent, so it is pooled
    # from every row of this dataset (one value per seed). The other five columns
    # have no clean-model counterpart, so they are structurally blank.
    baseline = cell(dataset_rows, "target_test_acc")
    body += _row("Baseline ($\\modelstd$)", [baseline, *([BLANK] * 5)])

    for epsilon in EPSILONS:
        eps_rows = by_epsilon.get(_epsilon_key(dataset_rows, epsilon), [])
        cells = [cell(eps_rows, column) for column in _EPS_COLUMNS]
        body += _row(f"$\\epsilon_{{rob}}$ = {epsilon:g}", cells)

    return body


def _epsilon_key(rows: Sequence[Mapping[str, str]], epsilon: float) -> str:
    """Return the CSV `epsilon` string matching this budget, or a miss sentinel.

    The CSV stores epsilon as text (`"0.1"`, `"0.10"`, ...), so a float budget is
    matched by value rather than by string identity.
    """
    for row in rows:
        try:
            if float(row["epsilon"]) == epsilon:
                return row["epsilon"]
        except ValueError:
            continue
    return f"__missing_{epsilon:g}__"


def render_table(results_file: Path) -> str:
    """Render the whole table from the E2 result CSV.

    Args:
        results_file: Path to `e2_advtr_modext.csv`. May be absent, in which case
            every cell renders blank.

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
        results_file: Path to `e2_advtr_modext.csv`.

    Returns:
        One human-readable line per (dataset, budget) cell.
    """
    rows = read_rows(results_file)
    lines: list[str] = []
    for dataset in DATASETS:
        dataset_rows = _rows_for_dataset(rows, dataset)
        baseline_seeds = seed_count(dataset_rows)
        status = (
            f"{len(baseline_seeds)} seed(s) {baseline_seeds}"
            if baseline_seeds
            else "MISSING"
        )
        lines.append(f"  {dataset:>8} baseline: {status}")
        by_epsilon = group_by(dataset_rows, "epsilon")
        for epsilon in EPSILONS:
            eps_rows = by_epsilon.get(_epsilon_key(dataset_rows, epsilon), [])
            seeds = seed_count(eps_rows)
            status = f"{len(seeds)} seed(s) {seeds}" if seeds else "MISSING"
            lines.append(f"  {dataset:>8} eps={epsilon:g}: {status}")
    return lines


def generate(
    results_dir: Path | None = None, out_dir: Path | None = None
) -> list[Path]:
    """Render the E2 table from a results base dir into a generated-output dir.

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
    """Return per-cell coverage lines for the E2 table from a results base dir."""
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
