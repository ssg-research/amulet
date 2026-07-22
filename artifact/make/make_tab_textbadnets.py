"""Render `tab_textbadnets_interactions` (E5) from the committed result CSVs.

    python artifact/make/make_tab_textbadnets.py

Reads `artifact/results/e5_textbadnets/{onion,dp}.csv` and writes
`artifact/tables/generated/tab_textbadnets_interactions.tex`. Rendering is a
pure function of those two files: no GPU, no model, no `llm` extra, seconds
(plan §13, decision 2).

**Seed-count agnostic.** A cell is aggregated over whatever seeds the CSVs
contain: five seeds render as `mean ~$\\pm$~ standard error`, one seed renders
as the bare value, which is exactly the shape the paper's DP-SGD block has. A
reviewer who re-runs one seed gets a one-seed table with the same structure,
never a crash and never a fabricated error bar.

**Output goes to `tables/generated/`, not over the reference table.** The
reference in `artifact/tables/` is the paper's own file and stays the fixed
point everything is compared against. Overwriting it in place would make the
comparison vacuous, and would destroy paper numbers the committed data does not
yet cover (the DP sweep is still running; see the P1 coverage report, which this
script re-prints on every run).
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import read_rows, results_path
from common.paths import artifact_root
from experiments.e5_textbadnets.schemas import DP_SCHEMA, ONION_SCHEMA

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from common.io import CsvSchema

EXPERIMENT_ID = "e5_textbadnets"
TABLE_STEM = "tab_textbadnets_interactions"

# A DP victim that predicts the target class for every input scores 100% ASR
# alongside the class prior (444 of SST-2's 872 validation records are positive,
# i.e. 50.9%). Such a cell measures the degenerate predictor rather than the
# defense, so the paper marks it and excludes it (§5, "Unintended Interaction").
COLLAPSE_MARKER = "$^{*}$"
_CHANCE_ACCURACY = 55.0

# Poison rates and privacy budgets the paper reports, used only to tell a reader
# which cells the committed CSVs do and do not cover.
PAPER_ONION_PORTIONS = (0.0001, 0.001, 0.01, 0.02, 0.05)
PAPER_DP_PORTIONS = (0.001, 0.01, 0.05)
PAPER_TARGET_EPSILONS = (1.0, 8.0)

_INDENT = " " * 8

_PREAMBLE = """\
\\begin{table*}[htb]
    \\setlength\\tabcolsep{4pt}
    \\centering
    \\footnotesize
    \\caption{Poisoning~(\\ref{poison}) versus two defenses on \\sstwo with a
    LoRA-adapted \\llama victim. Each block adds one defense to the poisoned model
    $\\modelpois$; both share the clean baseline $\\modelstd$. ONION drives $ASR$ down
    at every poison rate, while DP-SGD suppresses the backdoor only at small poison
    counts. $^{*}$This cell collapsed to the majority class and does not measure the
    defense.}%
    \\label{tab:textbadnets_interactions}
    \\begin{tabular}{ c c c c c c c }
        \\bottomrule

        \\toprule
        \\multirow{2}{*}{\\textbf{Poisoned}} & \\multirow{2}{*}{\\textbf{\\# Poisoned}} & \\multirow{2}{*}{\\textbf{Defense}} & \\multicolumn{2}{c}{\\textbf{Undefended} ($\\modelpois$)} & \\multicolumn{2}{c}{\\textbf{Defended} ($\\modeldef$)} \\\\
        & & & $Acc_{te}$ & $ASR$ & $Acc_{te}$ & $ASR$ \\\\
        \\midrule
"""

_EPILOGUE = """\
        \\bottomrule

        \\toprule
    \\end{tabular}
\\end{table*}
"""


def mean_and_standard_error(values: Sequence[float]) -> tuple[float, float]:
    """Return the mean of `values` and the standard error of that mean.

    Args:
        values: One measurement per seed. At least one.

    Returns:
        `(mean, standard_error)`. The standard error of a single measurement is
        zero: there is no spread to report, and the caller renders the value on
        its own rather than inventing an error bar.

    Raises:
        ValueError: If `values` is empty.
    """
    if not values:
        raise ValueError("Cannot aggregate an empty cell.")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, 0.0
    return mean, statistics.stdev(values) / len(values) ** 0.5


def format_cell(values: Sequence[float], marker: str = "") -> str:
    """Render one table cell from the per-seed measurements behind it.

    Args:
        values: One measurement per seed.
        marker: Footnote marker appended to the cell, e.g. the collapsed-cell
            asterisk.

    Returns:
        `"93.51~$\\pm$~0.17"` for several seeds, `"93.51"` for one.
    """
    mean, standard_error = mean_and_standard_error(values)
    if len(values) == 1:
        return f"{mean:.2f}{marker}"
    return f"{mean:.2f}~$\\pm$~{standard_error:.2f}{marker}"


def is_collapsed(row: Mapping[str, str]) -> bool:
    """Report whether a DP row is the degenerate majority-class predictor.

    Args:
        row: A row of the DP result CSV.

    Returns:
        True when the victim assigns the target class to every input: total
        attack success alongside chance accuracy.
    """
    return (
        float(row["dp_asr"]) >= 100.0 and float(row["dp_test_acc"]) <= _CHANCE_ACCURACY
    )


def aggregate_dp_cell(rows: Sequence[Mapping[str, str]]) -> tuple[str, str]:
    """Render the defended accuracy and ASR of one (poison rate, epsilon) cell.

    Collapsed seeds are excluded from the aggregate, as the paper excludes the
    collapsed configuration from its analysis: averaging a majority-class
    predictor into a healthy seed would report neither. When every seed of the
    cell collapsed there is nothing left to average, so the collapsed values are
    rendered as they are and marked with the footnote asterisk.

    Args:
        rows: The DP result rows of one cell, one per seed.

    Returns:
        `(accuracy_cell, asr_cell)`, already formatted for LaTeX.

    Raises:
        ValueError: If `rows` is empty.
    """
    if not rows:
        raise ValueError("Cannot aggregate a DP cell with no rows.")
    healthy = [row for row in rows if not is_collapsed(row)]
    marker = "" if healthy else COLLAPSE_MARKER
    used = healthy or list(rows)
    return (
        format_cell([float(row["dp_test_acc"]) for row in used], marker),
        format_cell([float(row["dp_asr"]) for row in used], marker),
    )


def _check_header(path: Path, schema: CsvSchema) -> None:
    """Fail loudly when a CSV's columns are not the ones this renderer reads."""
    rows = read_rows(path)
    if rows and tuple(rows[0]) != schema.header:
        raise ValueError(
            f"{path} does not carry the expected header. "
            f"Found: {', '.join(rows[0])}. Expected: {', '.join(schema.header)}."
        )


def _by_portion(
    rows: Iterable[Mapping[str, str]],
) -> dict[float, list[Mapping[str, str]]]:
    """Group result rows by poison rate, keyed by the rate as a float."""
    grouped: dict[float, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["poisoned_portion"])].append(row)
    return dict(sorted(grouped.items()))


def _per_seed_means(rows: Iterable[Mapping[str, str]], column: str) -> list[float]:
    """Collapse rows to one value per seed, averaging repeats within a seed.

    The seed is the unit of replication, so a seed that contributed several rows
    (E5 retrains and re-measures the clean baseline in every ONION run) must not
    weigh more heavily than one that contributed a single row.
    """
    by_seed: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_seed[row["exp_id"]].append(float(row[column]))
    return [statistics.fmean(values) for _, values in sorted(by_seed.items())]


def _row(cells: Sequence[str]) -> str:
    """Lay out one LaTeX table row at the table body's indentation."""
    return f"{_INDENT}{' & '.join(cells)} \\\\\n"


def percentage_label(portion: float) -> str:
    """Render a poison rate as the percentage the paper's first column shows."""
    text = f"{portion * 100:.2f}".rstrip("0").rstrip(".")
    return f"{text}\\%"


def _epsilon_label(target_epsilon: float) -> str:
    """Render a target privacy budget as the paper's `$\\epsilon = 8$`."""
    text = f"{target_epsilon:.2f}".rstrip("0").rstrip(".")
    return f"$\\epsilon = {text}$"


def _baseline_row(onion_rows: Sequence[Mapping[str, str]]) -> str:
    """Render the clean-baseline row shared by both blocks.

    Sourced from the ONION CSV alone, which sweeps every seed. The DP study
    reuses the *same* cached clean victim rather than training its own, so its
    `clean_baseline_test_acc` repeats a measurement the ONION rows already
    carry. Pooling both CSVs would therefore re-weight whichever seeds the DP
    sweep happens to have reached (currently seed 0 only) with a duplicate of
    one measurement, biasing the mean rather than adding evidence.
    """
    clean = _per_seed_means(onion_rows, "clean_baseline_test_acc")
    if not clean:
        return _row([
            "Baseline ($\\modelstd$)",
            "$-$",
            "$-$",
            "$-$",
            "$-$",
            "$-$",
            "$-$",
        ])
    return _row([
        "Baseline ($\\modelstd$)",
        "$-$",
        "$-$",
        format_cell(clean),
        "$-$",
        "$-$",
        "$-$",
    ])


def _onion_rows(rows: Sequence[Mapping[str, str]]) -> str:
    """Render the ONION block: one row per poison rate."""
    grouped = _by_portion(rows)
    lines: list[str] = []
    for index, (portion, cell_rows) in enumerate(grouped.items()):
        defense = f"\\multirow{{{len(grouped)}}}{{*}}{{ONION}}" if index == 0 else ""
        lines.append(
            _row([
                percentage_label(portion),
                cell_rows[0]["n_poisoned_train"],
                defense,
                format_cell(_per_seed_means(cell_rows, "undef_test_acc")),
                format_cell(_per_seed_means(cell_rows, "undef_asr")),
                format_cell(_per_seed_means(cell_rows, "def_test_acc_purified")),
                format_cell(_per_seed_means(cell_rows, "def_asr")),
            ])
        )
    return "".join(lines)


def _dp_rows(rows: Sequence[Mapping[str, str]]) -> str:
    """Render the DP-SGD block: one row per (poison rate, target epsilon)."""
    lines: list[str] = []
    for portion, rate_rows in _by_portion(rows).items():
        by_epsilon: dict[float, list[Mapping[str, str]]] = defaultdict(list)
        for row in rate_rows:
            by_epsilon[float(row["target_epsilon"])].append(row)
        span = len(by_epsilon)
        undefended = [
            f"\\multirow{{{span}}}{{*}}{{{format_cell(_per_seed_means(rate_rows, column))}}}"
            for column in ("undef_test_acc", "undef_asr")
        ]
        for index, (target_epsilon, cell_rows) in enumerate(sorted(by_epsilon.items())):
            accuracy, asr = aggregate_dp_cell(cell_rows)
            first = index == 0
            lines.append(
                _row([
                    f"\\multirow{{{span}}}{{*}}{{{percentage_label(portion)}}}"
                    if first
                    else "",
                    f"\\multirow{{{span}}}{{*}}{{{rate_rows[0]['n_poisoned_train']}}}"
                    if first
                    else "",
                    _epsilon_label(target_epsilon),
                    undefended[0] if first else "",
                    undefended[1] if first else "",
                    accuracy,
                    asr,
                ])
            )
    return "".join(lines)


def render_table(onion_csv: Path, dp_csv: Path) -> str:
    """Render the whole table from the two result CSVs.

    Args:
        onion_csv: Path to the ONION result CSV.
        dp_csv: Path to the DP-SGD result CSV. Either file may be absent or
            hold only a header, in which case its block is rendered empty.

    Returns:
        The `.tex` source, trailing-whitespace-free with a single final newline.
    """
    _check_header(onion_csv, ONION_SCHEMA)
    _check_header(dp_csv, DP_SCHEMA)
    onion_rows = read_rows(onion_csv)
    dp_rows = read_rows(dp_csv)

    body = [
        _baseline_row(onion_rows),
        f"{_INDENT}\\midrule\n",
        f"{_INDENT}\\multicolumn{{7}}{{c}}{{\\textbf{{ONION}} (intended interaction)}} \\\\\n",
        f"{_INDENT}\\midrule\n",
        _onion_rows(onion_rows),
        f"{_INDENT}\\midrule\n",
        f"{_INDENT}\\multicolumn{{7}}{{c}}{{\\textbf{{DP-SGD}} (unintended interaction)}} \\\\\n",
        f"{_INDENT}\\midrule\n",
        _dp_rows(dp_rows),
    ]
    return _PREAMBLE + "".join(body) + _EPILOGUE


def coverage(onion_csv: Path, dp_csv: Path) -> list[str]:
    """List the paper's cells the committed CSVs do not cover, one line each.

    Args:
        onion_csv: Path to the ONION result CSV.
        dp_csv: Path to the DP-SGD result CSV.

    Returns:
        Human-readable lines, empty when every paper cell is present. Seed
        counts are reported too: a cell backed by fewer seeds than the paper's
        five is covered, but its error bars are wider than the published ones.
    """
    lines: list[str] = []
    onion = _by_portion(read_rows(onion_csv))
    for portion in PAPER_ONION_PORTIONS:
        rows = onion.get(portion, [])
        seeds = sorted({row["exp_id"] for row in rows})
        lines.append(
            f"  ONION {percentage_label(portion):>8}: "
            + (f"{len(seeds)} seed(s) {seeds}" if seeds else "MISSING")
        )
    dp = _by_portion(read_rows(dp_csv))
    for portion in PAPER_DP_PORTIONS:
        for target_epsilon in PAPER_TARGET_EPSILONS:
            rows = [
                row
                for row in dp.get(portion, [])
                if float(row["target_epsilon"]) == target_epsilon
            ]
            seeds = sorted({row["exp_id"] for row in rows})
            lines.append(
                f"  DP-SGD {percentage_label(portion):>8} eps={target_epsilon:g}: "
                + (f"{len(seeds)} seed(s) {seeds}" if seeds else "MISSING")
            )
    return lines


def _csvs(results_dir: Path | None) -> tuple[Path, Path]:
    """Resolve the ONION and DP CSV paths under a results base dir."""
    return (
        results_path(EXPERIMENT_ID, "onion", base=results_dir),
        results_path(EXPERIMENT_ID, "dp", base=results_dir),
    )


def generate(
    results_dir: Path | None = None, out_dir: Path | None = None
) -> list[Path]:
    """Render the E5 table from a results base dir into a generated-output dir.

    Args:
        results_dir: Base directory holding `<experiment_id>/{onion,dp}.csv`.
            None reads the committed `results/`; a `runs/<level>/` directory
            renders a reviewer's re-run instead.
        out_dir: Directory the `.tex` is written to. None uses
            `tables/generated/`.

    Returns:
        The paths written (one `.tex`).
    """
    onion_csv, dp_csv = _csvs(results_dir)
    out_dir = artifact_root() / "tables" / "generated" if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{TABLE_STEM}.tex"
    output.write_text(render_table(onion_csv, dp_csv))
    return [output]


def coverage_report(results_dir: Path | None = None) -> list[str]:
    """Return per-cell coverage lines for the E5 table from a results base dir."""
    onion_csv, dp_csv = _csvs(results_dir)
    return coverage(onion_csv, dp_csv)


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
