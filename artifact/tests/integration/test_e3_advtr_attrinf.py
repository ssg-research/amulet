"""Tiny end-to-end run of E3, Adversarial Training x Attribute Inference (plan S8, L1).

At `test` level a small dense net over synthetic tabular rows with two sensitive
columns stands in for census/lfw, so the whole pipeline (split, train the clean
baseline, infer both attributes against it, adversarially train the defended
model, infer against it, append rows) runs on CPU in seconds with no download.
Assertions are "well-formed rows with in-range numbers, reproducibly", never
paper accuracy.

As in E2, the load-bearing test is that the defended model is genuinely the
adversarially-trained one, distinct from the clean target (plan S5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path

# Percentage metrics in [0, 100]; blank on the baseline row for the robust pair.
_PERCENT_COLUMNS = (
    "test_acc",
    "target_robust_acc",
    "defended_robust_acc",
    "acc_att_race",
    "acc_att_sex",
)
# AUC metrics in [0, 1].
_UNIT_COLUMNS = ("auc_race", "auc_sex")


def _context(tmp_path: Path, seed: int = 0):
    """Build a `test`-level run context over a throwaway cache."""
    from common.config import get_level
    from experiments import advtr_common as advtr

    config = get_level("test").with_defaults(epochs=100)
    torch.set_num_threads(1)
    advtr.seed_everything(seed)
    return advtr.RunContext(
        level=config, seed=seed, device="cpu", cache_dir=tmp_path / "models"
    )


def _in_range(row: dict[str, object]) -> None:
    """Assert every non-blank metric in a row is in its declared range."""
    for column in _PERCENT_COLUMNS:
        raw = row[column]
        if raw == "":
            continue
        value = cast(float, raw)
        assert 0.0 <= value <= 100.0, f"{column} outside [0, 100]: {value}"
    for column in _UNIT_COLUMNS:
        value = cast(float, row[column])
        assert 0.0 <= value <= 1.0, f"{column} outside [0, 1]: {value}"


@pytest.mark.integration
def test_test_level_run_writes_a_baseline_and_a_budget_row(tmp_path: Path) -> None:
    """One dataset produces a baseline row plus one row per budget, all in range."""
    from common.io import read_rows
    from experiments.e3_advtr_attrinf import run as e3

    rows = e3.run(
        level="test",
        seeds=(0,),
        datasets=("census",),
        epsilons=(0.01,),
        output_dir=tmp_path,
    )

    assert len(rows) == 2
    roles = {cast(str, row["model_role"]) for row in rows}
    assert roles == {"baseline", "defended"}
    for row in rows:
        _in_range(row)
    assert len(read_rows(tmp_path / "e3_advtr_attrinf.csv")) == 2


@pytest.mark.integration
def test_the_baseline_leaves_robust_columns_blank_and_budget_fills_them(
    tmp_path: Path,
) -> None:
    """Only the defended rows carry robust accuracies; the baseline's are blank."""
    from experiments.e3_advtr_attrinf import run as e3

    rows = e3.run(
        level="test",
        seeds=(0,),
        datasets=("census",),
        epsilons=(0.01,),
        output_dir=tmp_path,
    )
    baseline = next(r for r in rows if r["model_role"] == "baseline")
    defended = next(r for r in rows if r["model_role"] == "defended")

    assert baseline["target_robust_acc"] == ""
    assert baseline["defended_robust_acc"] == ""
    assert isinstance(defended["target_robust_acc"], float)
    assert isinstance(defended["defended_robust_acc"], float)


@pytest.mark.integration
def test_the_defended_model_is_not_the_clean_target(tmp_path: Path) -> None:
    """The defended model is a distinct, differently-trained network (plan S5).

    The old `advtr_attrinf.py` ran inference against the plain target whenever
    adversarial training was off; here the epsilon rows use a $\\modeldef$ that is
    a separate checkpoint with different weights from $\\modelstd$.
    """
    from experiments.e3_advtr_attrinf import run as e3

    ctx = _context(tmp_path)
    clean, split, data, clean_spec = e3.clean_target(ctx, "census")
    defended, defended_spec = e3.defended_target(ctx, "census", 0.01, split, data)

    assert clean is not defended
    assert clean_spec.key() != defended_spec.key()
    assert any(
        not torch.equal(a, b)
        for a, b in zip(
            clean.state_dict().values(), defended.state_dict().values(), strict=True
        )
    ), "adversarial training left the weights identical to the clean target"


@pytest.mark.integration
def test_the_clean_baseline_is_trained_once_across_budgets(tmp_path: Path) -> None:
    """Two budgets share one clean checkpoint: 1 clean + 2 defended = 3 files."""
    from experiments.e3_advtr_attrinf import run as e3

    _ = e3.run(
        level="test",
        seeds=(0,),
        datasets=("census",),
        epsilons=(0.01, 0.03),
        output_dir=tmp_path,
        cache_dir=tmp_path / "models",
    )

    assert len(sorted((tmp_path / "models").glob("*.pt"))) == 3


@pytest.mark.integration
def test_both_sensitive_attributes_are_scored(tmp_path: Path) -> None:
    """Every row carries an accuracy and an AUC for race and for sex."""
    from experiments.e3_advtr_attrinf import run as e3

    rows = e3.run(
        level="test",
        seeds=(0,),
        datasets=("census",),
        epsilons=(0.01,),
        output_dir=tmp_path,
    )

    for row in rows:
        assert isinstance(row["acc_att_race"], float)
        assert isinstance(row["acc_att_sex"], float)
        assert row["sensitive_attr_1"] == "attr_1"
        assert row["sensitive_attr_2"] == "attr_2"


@pytest.mark.integration
def test_rerunning_into_the_same_csv_does_not_duplicate(tmp_path: Path) -> None:
    """The idempotent append resumes an interrupted sweep rather than doubling it."""
    from common.io import read_rows
    from experiments.e3_advtr_attrinf import run as e3

    for _ in range(2):
        _ = e3.run(
            level="test",
            seeds=(0,),
            datasets=("census",),
            epsilons=(0.01,),
            output_dir=tmp_path,
            cache_dir=tmp_path / "models",
        )

    assert len(read_rows(tmp_path / "e3_advtr_attrinf.csv")) == 2


@pytest.mark.integration
def test_same_seed_reproduces(tmp_path: Path) -> None:
    """Two seeded runs agree within tolerance, so a reviewer's re-run is comparable."""
    from experiments.e3_advtr_attrinf import run as e3

    def once(where: Path) -> list[dict[str, object]]:
        return e3.run(
            level="test",
            seeds=(0,),
            datasets=("census",),
            epsilons=(0.01,),
            output_dir=where / "out",
            cache_dir=where / "models",
        )

    first = {r["model_role"]: r for r in once(tmp_path / "a")}
    second = {r["model_role"]: r for r in once(tmp_path / "b")}

    for role in ("baseline", "defended"):
        for column in ("acc_att_race", "acc_att_sex"):
            assert first[role][column] == pytest.approx(
                second[role][column], abs=5.0
            ), f"{role}.{column}"
        for column in _UNIT_COLUMNS:
            assert first[role][column] == pytest.approx(
                second[role][column], abs=0.1
            ), f"{role}.{column}"


@pytest.mark.integration
def test_a_test_level_run_leaves_the_committed_results_untouched(
    tmp_path: Path,
) -> None:
    """A tiny run must never write where the paper's numbers live."""
    from common.io import results_path
    from experiments.e3_advtr_attrinf import run as e3

    committed = results_path("e3_advtr_attrinf")
    before = committed.read_text() if committed.exists() else None

    _ = e3.run(level="test", seeds=(0,), datasets=("census",), epsilons=(0.01,))

    after = committed.read_text() if committed.exists() else None
    assert after == before
