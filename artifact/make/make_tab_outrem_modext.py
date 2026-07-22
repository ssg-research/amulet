"""Render `tab_outrem_modext` (E4) from the E4 result CSV.

    python artifact/make/make_tab_outrem_modext.py

Reads `artifact/runs/full/e4_outrem_modext.csv` and writes
`artifact/tables/generated/tab_outrem_modext.tex`. Rendering is a pure function
of the CSV: no GPU, no model, no download, seconds (plan S13, decision 2). The
figures share this same CSV through `make_fig_outrem.py`.

**Multi-dataset blocks.** One `\\multicolumn` block per dataset (census, lfw,
fmnist, cifar), each four rows deep ($Acc_{te}$ of $\\modeldef$, then the stolen
surrogate's $Acc_{te}$, $Fid$ and $Fid_{cor}$) across five removal columns: the
clean baseline $\\modelstd$ (removal 0) followed by 10/20/30/40%.

**Seed-count agnostic.** A cell aggregates over whatever seeds the CSV holds:
several as `mean ~$\\pm$~ SE`, one as the bare value, none as a dash.

**Output goes to `tables/generated/`.** This table is internal to the artifact:
the paper reports E4 as Figures 3 and 4, rendered by `make_fig_outrem.py`, so
there is no paper table to compare it against.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from make.tables_common import cell, group_by, seed_count

from common.io import read_rows, results_path
from common.paths import artifact_root
from experiments.e4_outrem_modext.schemas import (
    DATASETS,
    EXPERIMENT_ID,
    PERCENTS,
    SCHEMA,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

TABLE_STEM = "tab_outrem_modext"

_INDENT = " " * 4

# Dataset name -> the LaTeX macro the reference table uses for it.
_DATASET_MACRO: dict[str, str] = {
    "census": "\\census",
    "lfw": "\\lfw",
    "fmnist": "\\fmnist",
    "cifar": "\\cifar",
}

# The four data rows in each dataset block: (row label, CSV column).
_DATA_ROWS: tuple[tuple[str, str], ...] = (
    ("$Acc_{te}$", "defended_test_acc"),
    ("$\\modelstol$: $Acc_{te}$", "stolen_test_acc"),
    ("$\\modelstol$: $Fid$", "fidelity"),
    ("$\\modelstol$: $Fid_{cor}$", "correct_fidelity"),
)

_PREAMBLE = """\
\\begin{table*}[htbp]
\\footnotesize
\\caption{Interaction between~\\ref{outrem} Outlier Removal and~\\ref{modelext} Unauthorized Model Ownership. Removing outliers lowers $\\modeldef$ accuracy but leaves stolen-model fidelity essentially unchanged.}
\\label{tab:outrem_modext}
\\begin{center}
\\setlength\\tabcolsep{2pt}
\\begin{tabular}{ l c c c c c }
    \\bottomrule

    \\toprule
    & \\multirow{2}{*}{$\\modelstd$} & \\multicolumn{4}{c}{$\\modeldef$ (\\% of outliers removed followed by retraining)} \\\\
        & & \\textbf{10\\%} & \\textbf{20\\%} & \\textbf{30\\%} & \\textbf{40\\%} \\\\
    \\midrule
"""

_EPILOGUE = """\
    \\bottomrule

    \\toprule
\\end{tabular}
\\end{center}
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


def _percent_key(rows: Sequence[Mapping[str, str]], percent: int) -> str:
    """Return the CSV `percent` string matching this removal level, or a miss sentinel.

    The CSV stores percent as text (`"0"`, `"10"`, ...), so an int level is
    matched by value rather than by string identity.
    """
    for row in rows:
        try:
            if int(row["percent"]) == percent:
                return row["percent"]
        except ValueError:
            continue
    return f"__missing_{percent}__"


def _block(dataset: str, rows: Sequence[Mapping[str, str]]) -> str:
    """Render one dataset's header and its four metric rows across the removal grid."""
    dataset_rows = _rows_for_dataset(rows, dataset)
    by_percent = group_by(dataset_rows, "percent")

    body = f"{_INDENT}\\multicolumn{{6}}{{c}}{{\\textbf{{{_DATASET_MACRO[dataset]}}}}} \\\\\n"
    for label, column in _DATA_ROWS:
        cells = [
            cell(by_percent.get(_percent_key(dataset_rows, percent), []), column)
            for percent in PERCENTS
        ]
        body += _row(label, cells)
    return body


def render_table(results_file: Path) -> str:
    """Render the whole table from the E4 result CSV.

    Args:
        results_file: Path to `e4_outrem_modext.csv`. May be absent, in which
            case every cell renders blank.

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
    """List each cell's seed count, naming the removal levels with no data.

    Args:
        results_file: Path to `e4_outrem_modext.csv`.

    Returns:
        One human-readable line per (dataset, removal level) cell.
    """
    rows = read_rows(results_file)
    lines: list[str] = []
    for dataset in DATASETS:
        dataset_rows = _rows_for_dataset(rows, dataset)
        by_percent = group_by(dataset_rows, "percent")
        for percent in PERCENTS:
            pct_rows = by_percent.get(_percent_key(dataset_rows, percent), [])
            seeds = seed_count(pct_rows)
            status = f"{len(seeds)} seed(s) {seeds}" if seeds else "MISSING"
            label = "M_std" if percent == 0 else f"{percent}%"
            lines.append(f"  {dataset:>8} {label:>6}: {status}")
    return lines


def generate(
    results_dir: Path | None = None, out_dir: Path | None = None
) -> list[Path]:
    """Render the E4 table from a results base dir into a generated-output dir.

    Args:
        results_dir: Base directory holding the result CSV, in the shared
            layout. None reads `runs/full`; another `runs/<level>/`
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
    """Return per-cell coverage lines for the E4 table from a results base dir."""
    return coverage(results_path(EXPERIMENT_ID, base=results_dir))


def main() -> None:
    """Render the table from a results directory and report cell coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Base results directory to render from. Default: artifact/runs/full.",
    )
    args = parser.parse_args()

    for path in generate(results_dir=args.results_dir):
        print(f"wrote {path}")
    print("cell coverage:")
    for line in coverage_report(results_dir=args.results_dir):
        print(line)


if __name__ == "__main__":
    main()
