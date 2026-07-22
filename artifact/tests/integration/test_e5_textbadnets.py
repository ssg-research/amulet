"""Tiny end-to-end run of E5, the text-backdoor experiment (plan §8, Level 1).

The `test` level substitutes a two-layer randomly initialised Llama for the
3B LoRA target and eight synthetic SST-2-shaped sentences for the real corpus,
so the whole pipeline — poison, purify or privatise, train, score — runs on CPU
in seconds with no network access. What is asserted is "the experiment produces
a well-formed row with finite, in-range numbers, reproducibly", never paper
accuracy.

The tiny target is still a real `transformers` `LlamaModel`, so this module
needs the optional `llm` extra. The fast CI tier does not install it, hence the
module-level `importorskip`: without the extra these skip rather than fail.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip(
    "transformers",
    reason="E5 needs the optional `llm` extra: uv sync --extra cpu --extra llm",
)

ACCURACY_COLUMNS = {
    "onion": (
        "clean_baseline_test_acc",
        "undef_test_acc",
        "undef_asr",
        "def_test_acc_purified",
        "def_test_acc_raw",
        "def_asr",
    ),
    "dp": (
        "clean_baseline_test_acc",
        "undef_test_acc",
        "undef_asr",
        "dp_test_acc",
        "dp_asr",
    ),
}


@pytest.mark.integration
@pytest.mark.parametrize("which", ["onion", "dp"])
def test_test_level_run_writes_an_in_range_row(tmp_path: Path, which: str) -> None:
    """A tiny run produces one CSV row whose accuracies and ASRs are percentages."""
    from common.io import read_rows
    from experiments.e5_textbadnets import run as e5

    rows = e5.run(level="test", seeds=(0,), which=which, output_dir=tmp_path)

    assert len(rows) == 1
    for column in ACCURACY_COLUMNS[which]:
        value = cast(float, rows[0][column])
        assert 0.0 <= value <= 100.0, f"{column} outside [0, 100]: {value}"

    written = read_rows(tmp_path / f"{which}.csv")
    assert len(written) == 1
    assert float(written[0]["undef_asr"]) == pytest.approx(
        cast(float, rows[0]["undef_asr"])
    )


@pytest.mark.integration
def test_dp_run_reports_a_finite_spent_epsilon(tmp_path: Path) -> None:
    """DP-SGD's accountant returns a positive epsilon and noise multiplier."""
    from experiments.e5_textbadnets import run as e5

    rows = e5.run(level="test", seeds=(0,), which="dp", output_dir=tmp_path)

    assert cast(float, rows[0]["epsilon"]) > 0.0
    assert cast(float, rows[0]["sigma"]) > 0.0


@pytest.mark.integration
@pytest.mark.parametrize("which", ["onion", "dp"])
def test_same_seed_reproduces(tmp_path: Path, which: str) -> None:
    """Two runs of the same seed agree, so a reviewer's re-run is comparable."""
    from experiments.e5_textbadnets import run as e5

    first = e5.run(level="test", seeds=(0,), which=which, output_dir=tmp_path / "a")
    second = e5.run(level="test", seeds=(0,), which=which, output_dir=tmp_path / "b")

    for column in ACCURACY_COLUMNS[which]:
        assert first[0][column] == second[0][column], f"{column} not reproducible"


@pytest.mark.integration
def test_rerunning_into_the_same_csv_does_not_duplicate(tmp_path: Path) -> None:
    """The idempotent append means an interrupted sweep is resumed, not doubled."""
    from common.io import read_rows
    from experiments.e5_textbadnets import run as e5

    e5.run(level="test", seeds=(0,), which="onion", output_dir=tmp_path)
    e5.run(level="test", seeds=(0,), which="onion", output_dir=tmp_path)

    assert len(read_rows(tmp_path / "onion.csv")) == 1
