"""Render both E4 figures from the single committed result CSV.

    python artifact/make/make_fig_outrem.py

Reads `artifact/results/e4_outrem_modext.csv` and writes two figures into
`artifact/plots/generated/`, mirroring the `tables/generated/` convention for
tables (plan S7.1). Rendering is a pure function of the CSV: no GPU, no model,
no download, seconds (plan S13, decision 2). The table renderer
(`make_tab_outrem_modext.py`) reads the *same* CSV.

The two figures share one x-axis (percentage of outliers removed, `0` being the
clean baseline $\\modelstd$) and differ only in the solid series:

* `fig_outrem_fid`: dashed = $Acc_{te}$ of $\\modeldef$, solid = $Fid$ of the
  stolen model. Error bars from the per-seed standard error.
* `fig_outrem_cor_fid`: dashed = $Acc_{te}$, solid = $Fid_{cor}$. The reference
  omits error bars here for readability, so this figure does too.

Both are written as PNG (matching the committed reference format) and PDF (what
the paper's `\\includegraphics` pulls). A cell aggregates over whatever seeds the
CSV holds; a dataset with no rows is simply absent from the plot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # headless: no display, render straight to file.

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from make.tables_common import mean_and_standard_error, pooled_by_seed

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

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

PLOTS_STEM_FID = "fig_outrem_fid"
PLOTS_STEM_COR_FID = "fig_outrem_cor_fid"

# The column carrying the defended model's test accuracy (the dashed series in
# both figures).
_TEST_ACC_COLUMN = "defended_test_acc"

# Dataset -> colour, chosen to match the reference figures (census blue, lfw
# green, fmnist orange, cifar red). Cosmetic; the qualitative structure is what
# the reconstruction reproduces (plan S13.3).
_DATASET_COLOR: dict[str, str] = {
    "census": "C0",
    "lfw": "C2",
    "fmnist": "C1",
    "cifar": "C3",
}

# One aggregated series: the removal percentages present, and the per-percentage
# mean and standard error over seeds.
Series = tuple[list[int], list[float], list[float]]


def series_for(rows: Sequence[Mapping[str, str]], column: str) -> dict[str, Series]:
    """Aggregate one column into a per-dataset series over the removal grid.

    For each dataset, every removal percentage that carries at least one seed
    contributes a point: the mean over seeds and its standard error. Percentages
    with no data are omitted, so a partially-run sweep still plots what it has.

    Args:
        rows: The result rows (the whole CSV).
        column: The metric column to aggregate (e.g. `"fidelity"`).

    Returns:
        Dataset name -> `(percents, means, standard_errors)`, percents ascending.
        A dataset with no rows at all is absent from the mapping.
    """
    result: dict[str, Series] = {}
    for dataset in DATASETS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        percents: list[int] = []
        means: list[float] = []
        errors: list[float] = []
        for percent in PERCENTS:
            cell_rows = [row for row in dataset_rows if _percent_matches(row, percent)]
            values = pooled_by_seed(cell_rows, column)
            if not values:
                continue
            mean, error = mean_and_standard_error(values)
            percents.append(percent)
            means.append(mean)
            errors.append(error)
        if percents:
            result[dataset] = (percents, means, errors)
    return result


def _percent_matches(row: Mapping[str, str], percent: int) -> bool:
    """Report whether a row's `percent` cell equals this removal level by value."""
    try:
        return int(row["percent"]) == percent
    except (KeyError, ValueError):
        return False


def build_figure(
    rows: Sequence[Mapping[str, str]],
    *,
    solid_column: str,
    y_label: str,
    with_error_bars: bool,
) -> tuple[Figure, dict[str, dict[str, Series]]]:
    """Draw one E4 figure: dashed test accuracy and a solid metric, per dataset.

    Args:
        rows: The result rows (the whole CSV).
        solid_column: The metric drawn as the solid series (`"fidelity"` or
            `"correct_fidelity"`).
        y_label: The y-axis label.
        with_error_bars: Whether to draw per-seed standard-error bars.

    Returns:
        The figure, and the plotted data as
        `{dataset: {"test_acc": Series, solid_column: Series}}` so a test can
        check the aggregation without decoding pixels.
    """
    test_acc = series_for(rows, _TEST_ACC_COLUMN)
    solid = series_for(rows, solid_column)

    figure, axes = plt.subplots(figsize=(8, 5))
    plotted: dict[str, dict[str, Series]] = {}

    for dataset in DATASETS:
        color = _DATASET_COLOR[dataset]
        entry: dict[str, Series] = {}
        if dataset in solid:
            percents, means, errors = solid[dataset]
            axes.errorbar(
                percents,
                means,
                yerr=errors if with_error_bars else None,
                color=color,
                linestyle="-",
                marker="o",
                capsize=3,
            )
            entry[solid_column] = solid[dataset]
        if dataset in test_acc:
            percents, means, errors = test_acc[dataset]
            axes.errorbar(
                percents,
                means,
                yerr=errors if with_error_bars else None,
                color=color,
                linestyle="--",
                marker="x",
                capsize=3,
            )
            entry["test_acc"] = test_acc[dataset]
        if entry:
            plotted[dataset] = entry

    axes.set_xlabel("Outlier Removal (%)")
    axes.set_ylabel(y_label)
    axes.set_xticks(list(PERCENTS))
    axes.grid(True, linestyle="--", alpha=0.5)

    _attach_legends(axes, plotted)
    figure.tight_layout()
    return figure, plotted


def _attach_legends(axes: Axes, plotted: dict[str, dict[str, Series]]) -> None:
    """Attach the two reference legends: dataset colours, and line styles.

    Only datasets that actually plotted a series appear in the colour legend, so
    an empty figure carries no misleading entries.
    """
    color_handles = [
        Line2D([0], [0], color=_DATASET_COLOR[dataset], label=dataset)
        for dataset in DATASETS
        if dataset in plotted
    ]
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", marker="o", label="Fidelity"),
        Line2D(
            [0], [0], color="black", linestyle="--", marker="x", label="Test Accuracy"
        ),
    ]
    if color_handles:
        color_legend = axes.legend(handles=color_handles, loc="upper right")
        axes.add_artist(color_legend)
    axes.legend(handles=style_handles, loc="lower left")


def _save(figure: Figure, output_dir: Path, stem: str) -> list[Path]:
    """Write a figure as PNG and PDF, returning the paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        paths.append(path)
    return paths


def render_figures(results_file: Path, output_dir: Path) -> list[Path]:
    """Render both figures from the E4 result CSV into `output_dir`.

    Args:
        results_file: Path to `e4_outrem_modext.csv`. May be absent, in which
            case both figures render empty (axes and legends, no series).
        output_dir: Directory the figures are written to (created if absent).

    Returns:
        The paths written (PNG and PDF for each figure).

    Raises:
        ValueError: If the CSV's header does not match the schema.
    """
    rows = read_rows(results_file)
    if rows and tuple(rows[0]) != SCHEMA.header:
        raise ValueError(
            f"{results_file} does not carry the expected header. "
            f"Found: {', '.join(rows[0])}. Expected: {', '.join(SCHEMA.header)}."
        )

    written: list[Path] = []
    fid_figure, _ = build_figure(
        rows,
        solid_column="fidelity",
        y_label="Test Accuracy / Fidelity",
        with_error_bars=True,
    )
    written.extend(_save(fid_figure, output_dir, PLOTS_STEM_FID))
    plt.close(fid_figure)

    cor_fid_figure, _ = build_figure(
        rows,
        solid_column="correct_fidelity",
        y_label="Test Accuracy / Correct Fidelity",
        with_error_bars=False,
    )
    written.extend(_save(cor_fid_figure, output_dir, PLOTS_STEM_COR_FID))
    plt.close(cor_fid_figure)

    return written


def coverage(results_file: Path) -> list[str]:
    """List, per dataset, which removal levels carry data and which are missing."""
    rows = read_rows(results_file)
    fidelity = series_for(rows, "fidelity")
    lines: list[str] = []
    for dataset in DATASETS:
        present = fidelity.get(dataset, ([], [], []))[0]
        if present:
            covered = ", ".join(f"{p}%" for p in present)
            missing = [f"{p}%" for p in PERCENTS if p not in present]
            note = f"covered {covered}" + (
                f"; MISSING {', '.join(missing)}" if missing else ""
            )
        else:
            note = "MISSING (no data)"
        lines.append(f"  {dataset:>8}: {note}")
    return lines


def generate(
    results_dir: Path | None = None, out_dir: Path | None = None
) -> list[Path]:
    """Render both E4 figures from a results base dir into a generated-output dir.

    Args:
        results_dir: Base directory holding the result CSV, in the shared
            layout. None reads the committed `results/`; a `runs/<level>/`
            directory renders a reviewer's re-run instead.
        out_dir: Directory the figures are written to. None uses
            `plots/generated/`.

    Returns:
        The paths written (PNG and PDF for each of the two figures).
    """
    results_file = results_path(EXPERIMENT_ID, base=results_dir)
    out_dir = artifact_root() / "plots" / "generated" if out_dir is None else out_dir
    return render_figures(results_file, out_dir)


def coverage_report(results_dir: Path | None = None) -> list[str]:
    """Return per-dataset coverage lines for the figures from a results base dir."""
    return coverage(results_path(EXPERIMENT_ID, base=results_dir))


def main() -> None:
    """Render both figures from a results directory and report dataset coverage."""
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
    print("dataset coverage:")
    for line in coverage_report(results_dir=args.results_dir):
        print(line)


if __name__ == "__main__":
    main()
