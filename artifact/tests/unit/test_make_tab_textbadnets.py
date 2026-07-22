"""Contract for the E5 table renderer (plan §8, Level 1 unit tier).

Rendering is a pure function of the two E5 result CSVs, so every case
here is a hand-checked tiny CSV in, an exact `.tex` string out — no GPU, no
model, no `llm` extra. The numbers below are chosen so each mean and standard
error is checkable by eye: a two-seed cell of {90, 94} has mean 92.00 and
standard error 2.00.
"""

from __future__ import annotations

import csv
import re
from typing import TYPE_CHECKING

import pytest
from make.make_tab_textbadnets import aggregate_dp_cell, render_table

from experiments.e5_textbadnets.schemas import DP_SCHEMA, ONION_SCHEMA

if TYPE_CHECKING:
    from pathlib import Path

    from common.io import CsvSchema

# Columns the renderer never reads, filled with one plausible constant so the
# fixtures carry the real header without burying the numbers under test.
_ONION_CONSTANTS: dict[str, object] = {
    "dataset": "sst2",
    "model_name": "meta-llama/Llama-3.2-3B",
    "reference_model": "meta-llama/Llama-3.2-3B",
    "dtype": "float32",
    "num_classes": 2,
    "max_length": 128,
    "n_train": 67349,
    "clean_test_size": 872,
    "asr_test_size": 428,
    "batch_size": 16,
    "epochs": 3,
    "lr": 0.0002,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "trigger": "cf",
    "trigger_label": 1,
    "insert_position": "random",
    "onion_threshold": 0.0,
    "def_test_acc_raw": 95.0,
    "trigger_removal_rate": 1.0,
    "mean_words_removed": 2.5,
    "clean_train_runtime_sec": 1.0,
    "undef_train_runtime_sec": 1.0,
    "def_train_runtime_sec": 1.0,
    "onion_purify_runtime_sec": 1.0,
    "timestamp": "2026-07-21T00:00:00",
}

_DP_CONSTANTS: dict[str, object] = {
    "dataset": "sst2",
    "model_name": "meta-llama/Llama-3.2-3B",
    "dtype": "float32",
    "num_classes": 2,
    "max_length": 128,
    "n_train": 67349,
    "clean_test_size": 872,
    "asr_test_size": 428,
    "batch_size": 16,
    "epochs": 3,
    "lr": 0.0002,
    "dp_epochs": 3,
    "dp_lr": 0.0002,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "trigger": "cf",
    "trigger_label": 1,
    "insert_position": "random",
    "epsilon": 0.99,
    "sigma": 1.5,
    "delta": 1e-05,
    "max_per_sample_grad_norm": 1.0,
    "clean_train_runtime_sec": 1.0,
    "undef_train_runtime_sec": 1.0,
    "dp_train_runtime_sec": 1.0,
    "timestamp": "2026-07-21T00:00:00",
}

_HEADER_LINES = """\
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

_FOOTER_LINES = """\
        \\bottomrule

        \\toprule
    \\end{tabular}
\\end{table*}
"""


def _write(path: Path, schema: CsvSchema, rows: list[dict[str, object]]) -> Path:
    """Write `rows` as a CSV carrying the schema's full header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=schema.header)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _onion_row(
    seed: int,
    portion: float,
    n_poisoned: int,
    clean: float,
    undef_acc: float,
    undef_asr: float,
    def_acc: float,
    def_asr: float,
) -> dict[str, object]:
    return {
        **_ONION_CONSTANTS,
        "exp_id": seed,
        "poisoned_portion": portion,
        "n_poisoned_train": n_poisoned,
        "clean_baseline_test_acc": clean,
        "undef_test_acc": undef_acc,
        "undef_asr": undef_asr,
        "def_test_acc_purified": def_acc,
        "def_asr": def_asr,
    }


def _dp_row(
    seed: int,
    portion: float,
    n_poisoned: int,
    target_epsilon: float,
    clean: float,
    undef_acc: float,
    undef_asr: float,
    dp_acc: float,
    dp_asr: float,
) -> dict[str, object]:
    return {
        **_DP_CONSTANTS,
        "exp_id": seed,
        "poisoned_portion": portion,
        "n_poisoned_train": n_poisoned,
        "target_epsilon": target_epsilon,
        "clean_baseline_test_acc": clean,
        "undef_test_acc": undef_acc,
        "undef_asr": undef_asr,
        "dp_test_acc": dp_acc,
        "dp_asr": dp_asr,
    }


@pytest.fixture
def two_seed_csvs(tmp_path: Path) -> tuple[Path, Path]:
    """Two seeds x two poison rates for each study, with one collapsed DP cell."""
    onion = _write(
        tmp_path / "onion.csv",
        ONION_SCHEMA,
        [
            _onion_row(0, 0.0001, 6, 96.0, 90.0, 20.0, 80.0, 10.0),
            _onion_row(1, 0.0001, 6, 95.0, 94.0, 30.0, 84.0, 12.0),
            _onion_row(0, 0.01, 673, 96.0, 95.0, 100.0, 93.0, 8.0),
            _onion_row(1, 0.01, 673, 95.0, 96.0, 100.0, 94.0, 6.0),
        ],
    )
    dp = _write(
        tmp_path / "dp.csv",
        DP_SCHEMA,
        [
            _dp_row(0, 0.001, 67, 1.0, 96.0, 94.0, 99.0, 70.0, 28.0),
            _dp_row(1, 0.001, 67, 1.0, 95.0, 96.0, 99.0, 72.0, 30.0),
            _dp_row(0, 0.001, 67, 8.0, 96.0, 94.0, 99.0, 74.0, 32.0),
            _dp_row(1, 0.001, 67, 8.0, 95.0, 96.0, 99.0, 76.0, 34.0),
            # epsilon = 1 at 5% collapsed to the target class in both seeds.
            _dp_row(0, 0.05, 3367, 1.0, 96.0, 93.0, 100.0, 50.92, 100.0),
            _dp_row(1, 0.05, 3367, 1.0, 95.0, 93.0, 100.0, 50.92, 100.0),
            _dp_row(0, 0.05, 3367, 8.0, 96.0, 93.0, 100.0, 68.0, 96.0),
            _dp_row(1, 0.05, 3367, 8.0, 95.0, 93.0, 100.0, 70.0, 98.0),
        ],
    )
    return onion, dp


def test_renders_mean_and_standard_error_over_the_seeds_present(
    two_seed_csvs: tuple[Path, Path],
) -> None:
    """Two seeds per cell render as `mean ~$\\pm$~ standard error`."""
    onion, dp = two_seed_csvs

    rendered = render_table(onion, dp)

    assert (
        rendered
        == _HEADER_LINES
        + (
            "        Baseline ($\\modelstd$) & $-$ & $-$ & 95.50~$\\pm$~0.50 & $-$ & $-$ & $-$ \\\\\n"
            "        \\midrule\n"
            "        \\multicolumn{7}{c}{\\textbf{ONION} (intended interaction)} \\\\\n"
            "        \\midrule\n"
            "        0.01\\% & 6 & \\multirow{2}{*}{ONION} & 92.00~$\\pm$~2.00 & 25.00~$\\pm$~5.00 & 82.00~$\\pm$~2.00 & 11.00~$\\pm$~1.00 \\\\\n"
            "        1\\% & 673 &  & 95.50~$\\pm$~0.50 & 100.00~$\\pm$~0.00 & 93.50~$\\pm$~0.50 & 7.00~$\\pm$~1.00 \\\\\n"
            "        \\midrule\n"
            "        \\multicolumn{7}{c}{\\textbf{DP-SGD} (unintended interaction)} \\\\\n"
            "        \\midrule\n"
            "        \\multirow{2}{*}{0.1\\%} & \\multirow{2}{*}{67} & $\\epsilon = 1$ & \\multirow{2}{*}{95.00~$\\pm$~1.00} & \\multirow{2}{*}{99.00~$\\pm$~0.00} & 71.00~$\\pm$~1.00 & 29.00~$\\pm$~1.00 \\\\\n"
            "         &  & $\\epsilon = 8$ &  &  & 75.00~$\\pm$~1.00 & 33.00~$\\pm$~1.00 \\\\\n"
            "        \\multirow{2}{*}{5\\%} & \\multirow{2}{*}{3367} & $\\epsilon = 1$ & \\multirow{2}{*}{93.00~$\\pm$~0.00} & \\multirow{2}{*}{100.00~$\\pm$~0.00} & 50.92~$\\pm$~0.00$^{*}$ & 100.00~$\\pm$~0.00$^{*}$ \\\\\n"
            "         &  & $\\epsilon = 8$ &  &  & 69.00~$\\pm$~1.00 & 97.00~$\\pm$~1.00 \\\\\n"
        )
        + _FOOTER_LINES
    )


def test_single_seed_cells_render_without_an_error_term(tmp_path: Path) -> None:
    """One seed has no standard error, so the cell is the bare value.

    This is the shape of the paper's DP-SGD block, and it is what a reviewer's
    one-seed re-run produces: the renderer never assumes five seeds.
    """
    onion = _write(
        tmp_path / "onion.csv",
        ONION_SCHEMA,
        [_onion_row(0, 0.01, 673, 96.0, 95.0, 100.0, 93.0, 8.0)],
    )
    dp = _write(
        tmp_path / "dp.csv",
        DP_SCHEMA,
        [_dp_row(0, 0.05, 3367, 1.0, 96.0, 93.0, 100.0, 50.92, 100.0)],
    )

    rendered = render_table(onion, dp)

    assert (
        rendered
        == _HEADER_LINES
        + (
            "        Baseline ($\\modelstd$) & $-$ & $-$ & 96.00 & $-$ & $-$ & $-$ \\\\\n"
            "        \\midrule\n"
            "        \\multicolumn{7}{c}{\\textbf{ONION} (intended interaction)} \\\\\n"
            "        \\midrule\n"
            "        1\\% & 673 & \\multirow{1}{*}{ONION} & 95.00 & 100.00 & 93.00 & 8.00 \\\\\n"
            "        \\midrule\n"
            "        \\multicolumn{7}{c}{\\textbf{DP-SGD} (unintended interaction)} \\\\\n"
            "        \\midrule\n"
            "        \\multirow{1}{*}{5\\%} & \\multirow{1}{*}{3367} & $\\epsilon = 1$ & \\multirow{1}{*}{93.00} & \\multirow{1}{*}{100.00} & 50.92$^{*}$ & 100.00$^{*}$ \\\\\n"
        )
        + _FOOTER_LINES
    )


def test_a_collapsed_seed_is_dropped_when_the_cell_has_a_healthy_one() -> None:
    """A cell mixing a collapsed and a healthy seed aggregates the healthy one.

    The paper excludes the collapsed configuration from its analysis because it
    measures the majority-class predictor, not the defense. Averaging it into a
    healthy seed would import that artefact into the reported number.
    """
    collapsed = {"dp_test_acc": "50.92", "dp_asr": "100.0"}
    healthy = {"dp_test_acc": "68.00", "dp_asr": "96.00"}

    assert aggregate_dp_cell([collapsed, healthy]) == ("68.00", "96.00")


def test_an_all_collapsed_cell_is_rendered_and_marked() -> None:
    """With no healthy seed left, the collapsed values are shown with the marker."""
    collapsed = {"dp_test_acc": "50.92", "dp_asr": "100.0"}

    assert aggregate_dp_cell([collapsed]) == ("50.92$^{*}$", "100.00$^{*}$")


def test_an_unbalanced_dp_sweep_does_not_shift_the_baseline(tmp_path: Path) -> None:
    """The clean baseline comes from the ONION sweep alone.

    DP reuses the same cached clean target rather than training its own, and its
    sweep reaches fewer seeds than ONION (currently seed 0 only). Pooling both
    CSVs would re-weight seed 0 with a duplicate of a measurement the ONION rows
    already carry, biasing the mean rather than adding evidence, so a DP
    contribution -- however extreme -- must leave the baseline untouched.
    """
    onion = _write(
        tmp_path / "onion.csv",
        ONION_SCHEMA,
        [
            _onion_row(0, 0.01, 673, 96.0, 95.0, 100.0, 93.0, 8.0),
            _onion_row(1, 0.01, 673, 95.0, 96.0, 100.0, 94.0, 6.0),
        ],
    )
    # Seed 0 only, with a clean value far from either ONION baseline: pooling
    # would drag the rendered baseline from 95.50~$\pm$~0.50 to 84.00~$\pm$~11.00.
    dp = _write(
        tmp_path / "dp.csv",
        DP_SCHEMA,
        [_dp_row(0, 0.001, 67, 1.0, 50.0, 94.0, 99.0, 70.0, 28.0)],
    )

    rendered = render_table(onion, dp)

    assert (
        "        Baseline ($\\modelstd$) & $-$ & $-$ & 95.50~$\\pm$~0.50 & $-$ & $-$ & $-$ \\\\"
        in rendered.splitlines()
    )


_NUMBER = re.compile(r"\d+\.\d+|\d+")


def _body_numbers(latex: str) -> list[float]:
    """Return every number in a rendered table's `tabular` body.

    Restricted to the body, which excludes the caption and the length
    declarations preceding it. Formatting and caption wording are therefore free
    to change without failing a comparison; only the numbers are compared.
    """
    body = latex.split("\\begin{tabular}")[1].split("\\end{tabular}")[0]
    return [float(token) for token in _NUMBER.findall(body)]
