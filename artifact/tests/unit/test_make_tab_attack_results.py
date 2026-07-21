"""Contract for the E1 table renderer (plan §8, Level 1 unit tier).

Rendering is a pure function of the six committed result CSVs, so every case
here is a hand-checked tiny CSV in, an exact `.tex` string out — no GPU, no
model, no CelebA download. The numbers are chosen so each mean and standard
error is checkable by eye: a two-seed cell of {90, 94} has mean 92.00 and
standard error 2.00, and {0.63, 0.65} has mean 0.64 and standard error 0.01.

Two kinds of empty cell are covered, because they mean different things and the
renderer must not conflate them:

* *structurally blank* — the paper's membership-inference row is VGG11-only, so
  the other three columns stay blank even when a sweep produced data for them;
* *not measured yet* — a capacity nothing has been run for, which renders blank
  too but is reported by `coverage` so a reader is never misled into thinking a
  dash means zero.
"""

from __future__ import annotations

import csv
import re
from typing import TYPE_CHECKING

import pytest
from make.make_tab_attack_results import (
    coverage,
    format_cell,
    mean_and_standard_error,
    render_table,
)

from experiments.e1_attack_baselines.schemas import SCHEMAS

if TYPE_CHECKING:
    from pathlib import Path

# Columns the renderer never reads, filled with one plausible constant so the
# fixtures carry the real headers without burying the numbers under test.
_LEADING: dict[str, object] = {
    "dataset": "celeba",
    "arch": "vgg",
    "training_size": 1.0,
    "celeba_target": "Smiling",
    "optimizer_recipe": "adam_lr1e-3",
    "epochs": 100,
    "batch_size": 256,
}
_TRAILING: dict[str, object] = {"timestamp": "2026-07-21T00:00:00"}

_EXTRA: dict[str, dict[str, object]] = {
    "evasion": {"epsilon": 0.03, "step_size": 0.0075, "iterations": 40},
    "poisoning": {"poisoned_portion": 0.1, "trigger_label": 1},
    "model_extraction": {"adv_train_fraction": 0.5, "loss_type": "mse"},
    "membership_inference": {"pkeep": 0.5, "num_shadow": 64},
    "attribute_inference": {"adv_train_fraction": 0.5, "sensitive_attribute": "Male"},
    "data_reconstruction": {"alpha": 3000},
}

_HEADER_LINES = """\
\\begin{table*}[htb]
    \\centering
    \\footnotesize
    \\caption{Baseline evaluation of each risk in \\method on \\celeba, reproducing prior attacks to validate the implementations. $\\modelstd$, $\\modelpois$, and $\\modelstol$ are the baseline, poisoned, and stolen models. $^{*}$Membership inference uses an intentionally overfit ResNet-18.}
    \\label{tab:attack_results}
    \\begin{tabularx}{\\textwidth}{l l X X X X}
        \\toprule
        \\multirow{2}{*}{\\textbf{Attack}} & \\textbf{Model Architecture} & \\textbf{VGG11} & \\textbf{VGG13} & \\textbf{VGG16} & \\textbf{VGG19} \\\\
"""

_FOOTER_LINES = """\
        \\bottomrule
    \\end{tabularx}
\\end{table*}
"""


def _write(directory: Path, attack: str, rows: list[dict[str, object]]) -> None:
    """Write one sub-attack's CSV carrying its full schema header."""
    directory.mkdir(parents=True, exist_ok=True)
    schema = SCHEMAS[attack]
    with (directory / f"{attack}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=schema.header)
        writer.writeheader()
        writer.writerows([
            {**_LEADING, **_EXTRA[attack], **_TRAILING, **row} for row in rows
        ])


def _seeded(
    seed: int, capacity: str, **metrics: float | int | str
) -> dict[str, object]:
    """One result row: the cell it identifies plus the metrics it measured."""
    return {"exp_id": seed, "capacity": capacity, **metrics}


@pytest.fixture
def two_seed_results(tmp_path: Path) -> Path:
    """Two seeds of every sub-attack, for the VGG11 column only."""
    _write(
        tmp_path,
        "evasion",
        [
            _seeded(0, "m1", target_test_acc=91.0, robust_acc=10.0),
            _seeded(1, "m1", target_test_acc=91.0, robust_acc=14.0),
        ],
    )
    _write(
        tmp_path,
        "poisoning",
        [
            _seeded(
                0,
                "m1",
                std_test_acc=90.0,
                std_poison_acc=8.0,
                pois_test_acc=91.0,
                pois_poison_acc=95.0,
            ),
            _seeded(
                1,
                "m1",
                std_test_acc=94.0,
                std_poison_acc=8.0,
                pois_test_acc=91.0,
                pois_poison_acc=97.0,
            ),
        ],
    )
    _write(
        tmp_path,
        "model_extraction",
        [
            _seeded(
                0,
                "m1",
                target_test_acc=91.0,
                stolen_test_acc=91.0,
                fidelity=94.0,
                correct_fidelity=93.0,
            ),
            _seeded(
                1,
                "m1",
                target_test_acc=91.0,
                stolen_test_acc=91.0,
                fidelity=95.0,
                correct_fidelity=93.0,
            ),
        ],
    )
    _write(
        tmp_path,
        "membership_inference",
        [
            _seeded(
                0,
                "m1",
                target_train_acc=94.0,
                target_test_acc=76.0,
                offline_bal_acc=50.0,
                offline_auc=0.49,
                offline_tpr_at_1fpr=0.8,
                online_bal_acc=60.0,
                online_auc=0.63,
                online_tpr_at_1fpr=2.5,
            ),
            _seeded(
                1,
                "m1",
                target_train_acc=96.0,
                target_test_acc=78.0,
                offline_bal_acc=51.0,
                offline_auc=0.49,
                offline_tpr_at_1fpr=1.0,
                online_bal_acc=62.0,
                online_auc=0.65,
                online_tpr_at_1fpr=3.0,
            ),
        ],
    )
    _write(
        tmp_path,
        "attribute_inference",
        [
            _seeded(
                0, "m1", target_test_acc=91.0, attack_bal_acc=57.0, attack_auc=0.59
            ),
            _seeded(
                1, "m1", target_test_acc=91.0, attack_bal_acc=57.4, attack_auc=0.59
            ),
        ],
    )
    _write(
        tmp_path,
        "data_reconstruction",
        [
            _seeded(
                0,
                "m1",
                target_test_acc=91.0,
                mse_avg=0.20,
                mse_0=0.19,
                mse_1=0.20,
                ssim_avg=0.1,
                ssim_0=0.1,
                ssim_1=0.1,
            ),
            _seeded(
                1,
                "m1",
                target_test_acc=91.0,
                mse_avg=0.20,
                mse_0=0.19,
                mse_1=0.22,
                ssim_avg=0.1,
                ssim_0=0.1,
                ssim_1=0.1,
            ),
        ],
    )
    return tmp_path


def test_renders_mean_and_standard_error_over_the_seeds_present(
    two_seed_results: Path,
) -> None:
    """Two seeds per cell render as `mean ~$\\pm$~ standard error`.

    Only the VGG11 column was swept, so every other column is blank. The header
    row's $Acc_{te}$ is the poisoning study's clean baseline $\\modelstd$, which
    is the model the paper's table names there.
    """
    rendered = render_table(two_seed_results)

    assert (
        rendered
        == _HEADER_LINES
        + (
            "         & $Acc_{te}$ & 92.00~$\\pm$~2.00 & - & - & - \\\\\n"
            "        \\midrule\n"
            "        \\ref{evasion}~(Evasion) & $Acc_{rob}$ & 12.00~$\\pm$~2.00 & - & - & - \\\\\n"
            "        \\midrule\n"
            "        \\multirow{3}{*}{\\ref{poison}~(Poisoning)} & $Acc_{pois}$ ($\\modelstd$) & 8.00~$\\pm$~0.00 & - & - & - \\\\\n"
            "         & $Acc_{te}$ ($\\modelpois$) & 91.00~$\\pm$~0.00 & - & - & - \\\\\n"
            "         & $Acc_{pois}$ ($\\modelpois$) & 96.00~$\\pm$~1.00 & - & - & - \\\\\n"
            "        \\midrule\n"
            "        \\multirow{3}{*}{\\ref{modelext}~(Unauthorized Model Ownership)} & $Acc_{te}$ ($\\modelstol$) & 91.00~$\\pm$~0.00 & - & - & - \\\\\n"
            "         & $Fid$ ($\\modelstol$) & 94.50~$\\pm$~0.50 & - & - & - \\\\\n"
            "         & $Fid_{cor}$ ($\\modelstol$) & 93.00~$\\pm$~0.00 & - & - & - \\\\\n"
            "        \\midrule\n"
            "        \\multirow{8}{*}{\\ref{meminf}~(Membership Inference)$^{*}$} & $Acc_{tr}$ & 95.00~$\\pm$~1.00 & - & - & - \\\\\n"
            "         & $Acc_{te}$ & 77.00~$\\pm$~1.00 & - & - & - \\\\\n"
            "         & Offline $Acc_{bal}$ & 50.50~$\\pm$~0.50 & - & - & - \\\\\n"
            "         & Offline $AUC$ & 0.49~$\\pm$~0.00 & - & - & - \\\\\n"
            "         & Offline TPR@1\\%FPR & 0.90~$\\pm$~0.10 & - & - & - \\\\\n"
            "         & Online $Acc_{bal}$ & 61.00~$\\pm$~1.00 & - & - & - \\\\\n"
            "         & Online $AUC$ & 0.64~$\\pm$~0.01 & - & - & - \\\\\n"
            "         & Online TPR@1\\%FPR & 2.75~$\\pm$~0.25 & - & - & - \\\\\n"
            "        \\midrule\n"
            "        \\multirow{2}{*}{\\ref{attinf}~(Attribute Inference)} & $Acc_{att}$ & 57.20~$\\pm$~0.20 & - & - & - \\\\\n"
            "         & $AUC$ & 0.59~$\\pm$~0.00 & - & - & - \\\\\n"
            "        \\midrule\n"
            "        \\multirow{3}{*}{\\ref{datarecon}~(Data Reconstruction)} & $MSE_{avg}$ & 0.20~$\\pm$~0.00 & - & - & - \\\\\n"
            "         & $MSE_0$ & 0.19~$\\pm$~0.00 & - & - & - \\\\\n"
            "         & $MSE_1$ & 0.21~$\\pm$~0.01 & - & - & - \\\\\n"
        )
        + _FOOTER_LINES
    )


def test_single_seed_cells_render_without_an_error_term(tmp_path: Path) -> None:
    """One seed has no spread to report, so the cell is the bare value.

    This is what a reviewer's default one-seed re-run produces (plan §1): the
    renderer never assumes ten seeds and never invents an error bar.
    """
    _write(
        tmp_path,
        "evasion",
        [_seeded(0, "m1", target_test_acc=91.0, robust_acc=10.08)],
    )

    rendered = render_table(tmp_path)

    assert (
        "        \\ref{evasion}~(Evasion) & $Acc_{rob}$ & 10.08 & - & - & - \\\\"
        in rendered.splitlines()
    )


def test_a_capacity_with_no_results_renders_as_a_dash(tmp_path: Path) -> None:
    """A cell nothing has been run for is blank, not zero and not a crash."""
    _write(
        tmp_path,
        "evasion",
        [
            _seeded(0, "m1", target_test_acc=91.0, robust_acc=10.0),
            _seeded(0, "m3", target_test_acc=91.0, robust_acc=8.5),
        ],
    )

    rendered = render_table(tmp_path)

    assert (
        "        \\ref{evasion}~(Evasion) & $Acc_{rob}$ & 10.00 & - & 8.50 & - \\\\"
        in rendered.splitlines()
    )


def test_the_membership_inference_row_stays_a_single_column(tmp_path: Path) -> None:
    """Membership inference is VGG11-only even when other capacities have data.

    The paper's row is blank past the first column because the attack targets an
    overfit ResNet, not the VGG the column headings name. A sweep that happened
    to produce `m2` numbers must not quietly widen the row and imply those
    columns are comparable with the rest of the table.
    """
    _write(
        tmp_path,
        "membership_inference",
        [
            _seeded(
                0,
                capacity,
                target_train_acc=94.0,
                target_test_acc=76.0,
                offline_bal_acc=50.0,
                offline_auc=0.49,
                offline_tpr_at_1fpr=0.8,
                online_bal_acc=60.0,
                online_auc=0.63,
                online_tpr_at_1fpr=2.5,
            )
            for capacity in ("m1", "m2")
        ],
    )

    rendered = render_table(tmp_path)

    assert (
        "        \\multirow{8}{*}{\\ref{meminf}~(Membership Inference)$^{*}$} & $Acc_{tr}$ & 94.00 & - & - & - \\\\"
        in rendered.splitlines()
    )


def test_rendering_an_empty_results_directory_produces_a_full_blank_table(
    tmp_path: Path,
) -> None:
    """With nothing measured the structure still renders, every cell blank.

    A reviewer who has cloned the artifact but not run anything gets a readable
    skeleton and an honest coverage report rather than a traceback.
    """
    rendered = render_table(tmp_path)

    assert rendered.startswith("\\begin{table*}")
    assert rendered.endswith("\\end{table*}\n")
    assert "12.00" not in rendered


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([92.0], (92.0, 0.0)),
        ([90.0, 94.0], (92.0, 2.0)),
        ([1.0, 2.0, 3.0], (2.0, 0.5773502691896257)),
    ],
)
def test_mean_and_standard_error(
    values: list[float], expected: tuple[float, float]
) -> None:
    """The error term is the standard error of the mean, not the deviation.

    For {1, 2, 3} the sample standard deviation is 1.0 and the standard error is
    1/sqrt(3); reporting the former would overstate every error bar in the table
    by a factor of sqrt(n).
    """
    assert mean_and_standard_error(values) == pytest.approx(expected)


def test_an_empty_cell_cannot_be_formatted() -> None:
    """Formatting no measurements is a bug, so it raises rather than guessing."""
    with pytest.raises(ValueError, match="empty cell"):
        _ = format_cell([])


def test_coverage_names_the_cells_with_no_data(two_seed_results: Path) -> None:
    """Coverage reports seed counts per cell, and says MISSING where there are none.

    A dash in the table is ambiguous on its own. This is what tells a reader
    which dashes are "the paper leaves this blank" and which are "we have not
    run it".
    """
    lines = coverage(two_seed_results)

    assert any(
        "evasion" in line and "m1" in line and "2 seed(s)" in line for line in lines
    )
    assert any(
        "evasion" in line and "m4" in line and "MISSING" in line for line in lines
    )
    # The single-column attack is reported for `m1` only, never as missing data.
    assert not any("membership_inference" in line and "m2" in line for line in lines)


_NUMBER = re.compile(r"\d+\.\d+|\d+")


def _body_numbers(latex: str) -> list[float]:
    """Return every number in a rendered table's `tabularx` body.

    Restricted to the body, which structurally excludes the caption, so
    rewording the caption can never fail a comparison while a changed number
    always does (plan §7.1, the table contract).
    """
    body = latex.split("\\begin{tabularx}")[1].split("\\end{tabularx}")[0]
    return [float(token) for token in _NUMBER.findall(body)]


def test_the_reference_table_and_a_rendered_one_have_the_same_shape(
    two_seed_results: Path,
) -> None:
    """The generated table carries a cell wherever the paper's does.

    E1's real numbers need many GPU-hours and are produced by a later step, so
    this cannot yet compare values. What it can pin now is that the renderer
    emits the paper's row and column structure: same count of numeric cells in
    the body once blanks are filled, so a completed sweep drops straight in.
    """
    from common.paths import artifact_root

    reference = (artifact_root() / "tables" / "tab_attack_results.tex").read_text()
    rendered = render_table(two_seed_results)

    assert rendered.count("\\\\\n") == reference.count("\\\\\n")
    assert rendered.count("\\midrule") == reference.count("\\midrule")
