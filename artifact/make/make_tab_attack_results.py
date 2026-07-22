"""Render `tab_attack_results` (E1) from the committed result CSVs.

    python artifact/make/make_tab_attack_results.py

Reads `artifact/results/e1_attack_baselines/<attack>.csv` and writes
`artifact/tables/generated/tab_attack_results.tex`. Rendering is a pure function
of those files: no GPU, no model, no CelebA download, seconds (plan §13,
decision 2).

**Seed-count agnostic.** A cell is aggregated over whatever seeds the CSVs
contain: several seeds render as `mean ~$\\pm$~ standard error`, one seed as the
bare value. The paper reports ten repeats; a reviewer's single-seed re-run gets
a one-seed table of the same shape, never a crash and never a fabricated error
bar.

**Two kinds of empty cell.** A dash is emitted both where the paper's table is
structurally blank (the membership-inference row is VGG11-only) and where a
capacity simply has no data yet. The two are indistinguishable in the `.tex`, so
`coverage()` reports which cells are backed by data and which are missing, and
`main` prints it on every run.

**Output goes to `tables/generated/`.** The paper's own tables are not mirrored
in this repository, so the fixed point a rendered table is compared against is
the paper itself (Table 5).
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import read_rows, results_root
from common.paths import artifact_root
from experiments.e1_attack_baselines.schemas import (
    CAPACITIES,
    SCHEMAS,
    SINGLE_COLUMN_ATTACKS,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from common.io import CsvSchema

EXPERIMENT_ID = "e1_attack_baselines"
TABLE_STEM = "tab_attack_results"

_INDENT = " " * 8
_BLANK = "-"

_PREAMBLE = """\
\\begin{table*}[htb]
    \\centering
    \\footnotesize
    \\caption{Baseline evaluation of each risk in \\method on \\celeba, reproducing prior attacks to validate the implementations. $\\modelstd$, $\\modelpois$, and $\\modelstol$ are the baseline, poisoned, and stolen models. $^{*}$Membership inference uses an intentionally overfit ResNet-18.}
    \\label{tab:attack_results}
    \\begin{tabularx}{\\textwidth}{l l X X X X}
        \\toprule
        \\multirow{2}{*}{\\textbf{Attack}} & \\textbf{Model Architecture} & \\textbf{VGG11} & \\textbf{VGG13} & \\textbf{VGG16} & \\textbf{VGG19} \\\\
"""

_EPILOGUE = """\
        \\bottomrule
    \\end{tabularx}
\\end{table*}
"""


class MetricRow:
    """One printed line of the table: a metric label and the column it reads.

    Attributes:
        label: The row's metric label, the table's second column.
        attack: Which sub-attack CSV the value comes from.
        column: Which CSV column the value is.
        precision: Decimal places to render the mean and error to.
    """

    def __init__(
        self, label: str, attack: str, column: str, precision: int = 2
    ) -> None:
        self.label = label
        self.attack = attack
        self.column = column
        self.precision = precision


# The table's rows, in order, each naming the CSV column it renders. The
# grouping (how many rows sit under one `\multirow` attack label) is described
# by `_BLOCKS` below.
_HEADER_METRIC = MetricRow("$Acc_{te}$", "poisoning", "std_test_acc")

_EVASION = [MetricRow("$Acc_{rob}$", "evasion", "robust_acc")]

_POISONING = [
    MetricRow("$Acc_{pois}$ ($\\modelstd$)", "poisoning", "std_poison_acc"),
    MetricRow("$Acc_{te}$ ($\\modelpois$)", "poisoning", "pois_test_acc"),
    MetricRow("$Acc_{pois}$ ($\\modelpois$)", "poisoning", "pois_poison_acc"),
]

_MODEL_EXTRACTION = [
    MetricRow("$Acc_{te}$ ($\\modelstol$)", "model_extraction", "stolen_test_acc"),
    MetricRow("$Fid$ ($\\modelstol$)", "model_extraction", "fidelity"),
    MetricRow("$Fid_{cor}$ ($\\modelstol$)", "model_extraction", "correct_fidelity"),
]

_MEMBERSHIP_INFERENCE = [
    MetricRow("$Acc_{tr}$", "membership_inference", "target_train_acc"),
    MetricRow("$Acc_{te}$", "membership_inference", "target_test_acc"),
    MetricRow("Offline $Acc_{bal}$", "membership_inference", "offline_bal_acc"),
    MetricRow("Offline $AUC$", "membership_inference", "offline_auc"),
    MetricRow("Offline TPR@1\\%FPR", "membership_inference", "offline_tpr_at_1fpr"),
    MetricRow("Online $Acc_{bal}$", "membership_inference", "online_bal_acc"),
    MetricRow("Online $AUC$", "membership_inference", "online_auc"),
    MetricRow("Online TPR@1\\%FPR", "membership_inference", "online_tpr_at_1fpr"),
]

_ATTRIBUTE_INFERENCE = [
    MetricRow("$Acc_{att}$", "attribute_inference", "attack_bal_acc"),
    MetricRow("$AUC$", "attribute_inference", "attack_auc"),
]

_DATA_RECONSTRUCTION = [
    MetricRow("$MSE_{avg}$", "data_reconstruction", "mse_avg"),
    MetricRow("$MSE_0$", "data_reconstruction", "mse_0"),
    MetricRow("$MSE_1$", "data_reconstruction", "mse_1"),
]

# Each block is one attack label spanning its metric rows.
_BLOCKS: list[tuple[str, list[MetricRow]]] = [
    ("\\ref{evasion}~(Evasion)", _EVASION),
    ("\\ref{poison}~(Poisoning)", _POISONING),
    ("\\ref{modelext}~(Unauthorized Model Ownership)", _MODEL_EXTRACTION),
    ("\\ref{meminf}~(Membership Inference)$^{*}$", _MEMBERSHIP_INFERENCE),
    ("\\ref{attinf}~(Attribute Inference)", _ATTRIBUTE_INFERENCE),
    ("\\ref{datarecon}~(Data Reconstruction)", _DATA_RECONSTRUCTION),
]

# AUC metrics are fractions of one, rendered to two decimals like the paper; the
# rest are percentages. Both use two decimals, so precision is uniform, but the
# distinction is what would change were a metric ever reported differently.


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


def format_cell(values: Sequence[float], precision: int = 2) -> str:
    """Render one table cell from the per-seed measurements behind it.

    Args:
        values: One measurement per seed. At least one.
        precision: Decimal places for the mean and the error.

    Returns:
        `"57.17~$\\pm$~0.15"` for several seeds, `"57.17"` for one.

    Raises:
        ValueError: If `values` is empty.
    """
    mean, standard_error = mean_and_standard_error(values)
    if len(values) == 1:
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f}~$\\pm$~{standard_error:.{precision}f}"


def _by_capacity(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, list[Mapping[str, str]]]:
    """Group an attack's result rows by capacity column."""
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["capacity"]].append(row)
    return grouped


def _values_by_capacity(
    rows: Sequence[Mapping[str, str]], column: str
) -> dict[str, list[float]]:
    """Collect one value per seed for a column, keyed by capacity.

    The seed (`exp_id`) is the unit of replication, so a capacity that a seed
    contributed several rows to (which the idempotent append prevents, but a
    hand-edited CSV could hold) is de-duplicated to one value per seed.
    """
    by_capacity: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_capacity[row["capacity"]][row["exp_id"]] = float(row[column])
    return {
        capacity: [seeds[seed] for seed in sorted(seeds)]
        for capacity, seeds in by_capacity.items()
    }


def _rows_for(results_dir: Path, attack: str) -> list[dict[str, str]]:
    """Read one sub-attack's CSV, validating its header if present."""
    path = results_dir / f"{attack}.csv"
    rows = read_rows(path)
    schema: CsvSchema = SCHEMAS[attack]
    if rows and tuple(rows[0]) != schema.header:
        raise ValueError(
            f"{path} does not carry the expected header. "
            f"Found: {', '.join(rows[0])}. Expected: {', '.join(schema.header)}."
        )
    return rows


def _columns_for(attack: str) -> tuple[str, ...]:
    """Return the capacity columns an attack occupies in the table.

    A single-column attack (membership inference) occupies the first column
    only, so the rest of its row stays blank as in the paper regardless of what
    a sweep produced.
    """
    if attack in SINGLE_COLUMN_ATTACKS:
        return CAPACITIES[:1]
    return CAPACITIES


def _cells(metric: MetricRow, values: Mapping[str, list[float]]) -> list[str]:
    """Render a metric's four capacity cells, blanking those with no data."""
    columns = _columns_for(metric.attack)
    cells: list[str] = []
    for capacity in CAPACITIES:
        seeds = values.get(capacity, [])
        if capacity in columns and seeds:
            cells.append(format_cell(seeds, metric.precision))
        else:
            cells.append(_BLANK)
    return cells


def _metric_line(prefix: str, metric: MetricRow, cells: Sequence[str]) -> str:
    """Lay out one metric row: an attack-label column, the metric, the cells."""
    return f"{_INDENT}{prefix} & {metric.label} & {' & '.join(cells)} \\\\\n"


def render_table(results_dir: Path) -> str:
    """Render the whole table from an experiment's result directory.

    Args:
        results_dir: Directory holding `<attack>.csv` for each sub-attack. Any
            file may be absent, in which case its cells render blank.

    Returns:
        The `.tex` source, trailing-whitespace-free with a single final newline.
    """
    rows_by_attack = {attack: _rows_for(results_dir, attack) for attack in SCHEMAS}

    def cells_for(metric: MetricRow) -> list[str]:
        values = _values_by_capacity(rows_by_attack[metric.attack], metric.column)
        return _cells(metric, values)

    body: list[str] = []
    # The header line's own metric: the clean baseline's test accuracy.
    body.append(_metric_line("", _HEADER_METRIC, cells_for(_HEADER_METRIC)))

    for label, metrics in _BLOCKS:
        body.append(f"{_INDENT}\\midrule\n")
        span = len(metrics)
        for index, metric in enumerate(metrics):
            prefix = (
                f"\\multirow{{{span}}}{{*}}{{{label}}}"
                if index == 0 and span > 1
                else (label if span == 1 else "")
            )
            body.append(_metric_line(prefix, metric, cells_for(metric)))

    return _PREAMBLE + "".join(body) + _EPILOGUE


def coverage(results_dir: Path) -> list[str]:
    """List each cell's seed count, and name the ones with no data.

    A dash in the table is ambiguous on its own. This tells a reader which
    dashes are "the paper leaves this blank" (never listed as missing) and which
    are "this capacity has not been run" (listed as MISSING). Cells backed by
    fewer than the paper's ten seeds are covered but have wider error bars.

    Args:
        results_dir: Directory holding `<attack>.csv` for each sub-attack.

    Returns:
        One human-readable line per (attack, capacity) cell the table can hold.
    """
    lines: list[str] = []
    for attack in SCHEMAS:
        rows = _rows_for(results_dir, attack)
        by_capacity = _by_capacity(rows)
        for capacity in _columns_for(attack):
            seeds = sorted({row["exp_id"] for row in by_capacity.get(capacity, [])})
            status = f"{len(seeds)} seed(s) {seeds}" if seeds else "MISSING"
            lines.append(f"  {attack:>22} {capacity}: {status}")
    return lines


def _experiment_dir(results_dir: Path | None) -> Path:
    """Resolve E1's per-experiment subdirectory under a results base dir.

    E1 emits one CSV per sub-attack into a `<experiment_id>/` subdirectory, in
    both `results/` and `runs/<level>/`, so the base dir is joined with the
    experiment id to reach them.
    """
    base = results_root() if results_dir is None else results_dir
    return base / EXPERIMENT_ID


def generate(
    results_dir: Path | None = None, out_dir: Path | None = None
) -> list[Path]:
    """Render the E1 table from a results base dir into a generated-output dir.

    Args:
        results_dir: Base directory holding `<experiment_id>/<attack>.csv`. None
            reads the committed `results/`; a `runs/<level>/` directory renders a
            reviewer's re-run instead.
        out_dir: Directory the `.tex` is written to. None uses
            `tables/generated/`.

    Returns:
        The paths written (one `.tex`).
    """
    experiment_dir = _experiment_dir(results_dir)
    out_dir = artifact_root() / "tables" / "generated" if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{TABLE_STEM}.tex"
    output.write_text(render_table(experiment_dir))
    return [output]


def coverage_report(results_dir: Path | None = None) -> list[str]:
    """Return per-cell coverage lines for the E1 table from a results base dir."""
    return coverage(_experiment_dir(results_dir))


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
