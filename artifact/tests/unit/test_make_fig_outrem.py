"""Contract for the E4 plot renderer (plan S8 Level 1, S7.1, S13.3).

Both figures are a pure function of the committed CSV, so every case is a
hand-checked tiny CSV in, a figure (or a PNG file) out: no GPU, no model, no
download. The numeric contract is `series_for`, checked by eye; the drawing
contract is the axis labels, the two legends, and a non-empty PNG on disk.

E4 is a reconstruction with no source CSV, so these tests pin the figures'
*structure* (axis labels, legends, a non-empty PNG), never their pixel values.
The paper's own figures are not mirrored in this repository, so numeric
reproduction is a comparison against the paper's Figures 3 and 4 after an L3 run.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from make.make_fig_outrem import (
    PLOTS_STEM_COR_FID,
    PLOTS_STEM_FID,
    build_figure,
    render_figures,
    series_for,
)
from matplotlib.legend import Legend

from experiments.e4_outrem_modext.schemas import SCHEMA

if TYPE_CHECKING:
    from pathlib import Path

# Rows are string-valued, exactly as `csv.DictReader` hands them to the renderer
# from a real result CSV, so the in-memory fixtures type-check against the
# `Mapping[str, str]` the renderer reads.
_FILLER: dict[str, str] = {
    "arch": "linearnet",
    "capacity": "m1",
    "training_size": "1.0",
    "epochs": "100",
    "batch_size": "256",
    "adv_train_fraction": "0.5",
    "timestamp": "2026-07-21T00:00:00",
}


def _row(
    seed: int,
    dataset: str,
    percent: int,
    *,
    defended_test_acc: float,
    stolen_test_acc: float,
    fidelity: float,
    correct_fidelity: float,
) -> dict[str, str]:
    return {
        "exp_id": str(seed),
        "dataset": dataset,
        "percent": str(percent),
        "defended_test_acc": str(defended_test_acc),
        "stolen_test_acc": str(stolen_test_acc),
        "fidelity": str(fidelity),
        "correct_fidelity": str(correct_fidelity),
    }


def _write(directory: Path, rows: list[dict[str, str]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "e4_outrem_modext.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEMA.header)
        writer.writeheader()
        writer.writerows([{**_FILLER, **row} for row in rows])
    return path


def _all_four_datasets() -> list[dict[str, str]]:
    """One seed, two removal levels (0, 10), for all four datasets."""
    rows: list[dict[str, str]] = []
    for dataset in ("census", "lfw", "fmnist", "cifar"):
        for percent, acc, fid, cor in [(0, 84.0, 99.0, 84.0), (10, 81.0, 98.0, 81.0)]:
            rows.append(
                _row(
                    0,
                    dataset,
                    percent,
                    defended_test_acc=acc,
                    stolen_test_acc=acc,
                    fidelity=fid,
                    correct_fidelity=cor,
                )
            )
    return rows


def test_series_pools_the_mean_per_dataset_and_percent() -> None:
    """`series_for` returns one point per removal level: the mean over seeds.

    census fidelity at 10% over seeds {0, 1} = {98, 96} -> 97.00, standard
    error 1.00; the percents come back ascending.
    """
    rows = [
        _row(
            0,
            "census",
            10,
            defended_test_acc=81.0,
            stolen_test_acc=81.0,
            fidelity=98.0,
            correct_fidelity=81.0,
        ),
        _row(
            1,
            "census",
            10,
            defended_test_acc=83.0,
            stolen_test_acc=83.0,
            fidelity=96.0,
            correct_fidelity=83.0,
        ),
        _row(
            0,
            "census",
            0,
            defended_test_acc=84.0,
            stolen_test_acc=84.0,
            fidelity=99.0,
            correct_fidelity=84.0,
        ),
    ]

    series = series_for(rows, "fidelity")

    percents, means, errors = series["census"]
    assert percents == [0, 10]
    assert means == pytest.approx([99.0, 97.0])
    assert errors == pytest.approx([0.0, 1.0])


def test_a_dataset_with_no_rows_is_absent_from_the_series() -> None:
    """A dataset nothing has been run for contributes no line."""
    rows = [
        _row(
            0,
            "census",
            0,
            defended_test_acc=84.0,
            stolen_test_acc=84.0,
            fidelity=99.0,
            correct_fidelity=84.0,
        )
    ]

    series = series_for(rows, "fidelity")

    assert set(series) == {"census"}


def test_the_fid_figure_has_the_reference_axis_labels(tmp_path: Path) -> None:
    """fig_outrem_fid's axes read 'Outlier Removal (%)' and 'Test Accuracy / Fidelity'."""
    figure, _ = build_figure(
        _all_four_datasets(),
        solid_column="fidelity",
        y_label="Test Accuracy / Fidelity",
        with_error_bars=True,
    )
    axes = figure.axes[0]

    assert axes.get_xlabel() == "Outlier Removal (%)"
    assert axes.get_ylabel() == "Test Accuracy / Fidelity"
    plt.close(figure)


def test_the_figure_carries_a_dataset_legend_and_a_style_legend() -> None:
    """Two legends: the four dataset colours, and the Fidelity/Test-Accuracy styles."""
    figure, _ = build_figure(
        _all_four_datasets(),
        solid_column="fidelity",
        y_label="Test Accuracy / Fidelity",
        with_error_bars=True,
    )
    axes = figure.axes[0]

    legends = [child for child in axes.get_children() if isinstance(child, Legend)]
    labels = {text.get_text() for legend in legends for text in legend.get_texts()}

    assert len(legends) == 2
    assert {"census", "lfw", "fmnist", "cifar"} <= labels
    assert {"Fidelity", "Test Accuracy"} <= labels
    plt.close(figure)


def test_each_dataset_draws_a_test_accuracy_and_a_metric_series() -> None:
    """Every plotted dataset carries both a dashed test-acc and a solid metric series."""
    _, plotted = build_figure(
        _all_four_datasets(),
        solid_column="fidelity",
        y_label="Test Accuracy / Fidelity",
        with_error_bars=True,
    )

    assert set(plotted) == {"census", "lfw", "fmnist", "cifar"}
    for dataset in plotted:
        assert set(plotted[dataset]) == {"test_acc", "fidelity"}


def test_render_figures_writes_two_non_empty_pngs(tmp_path: Path) -> None:
    """Both reference figures land on disk as non-empty PNGs (plus PDFs)."""
    csv_path = _write(tmp_path / "results", _all_four_datasets())
    out = tmp_path / "generated"

    written = render_figures(csv_path, out)

    for stem in (PLOTS_STEM_FID, PLOTS_STEM_COR_FID):
        png = out / f"{stem}.png"
        assert png in written
        assert png.stat().st_size > 0
        assert (out / f"{stem}.pdf") in written


def test_render_figures_on_an_absent_csv_still_draws_empty_figures(
    tmp_path: Path,
) -> None:
    """With nothing measured the figures still render (axes, no series), never crash."""
    out = tmp_path / "generated"

    written = render_figures(tmp_path / "does_not_exist.csv", out)

    assert (out / f"{PLOTS_STEM_FID}.png").stat().st_size > 0
    assert len(written) == 4
