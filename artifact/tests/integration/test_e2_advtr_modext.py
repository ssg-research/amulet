"""Tiny end-to-end run of E2, Adversarial Training x Model Ownership (plan S8, L1).

At `test` level a small dense net over a handful of synthetic tabular rows
stands in for every dataset, so the whole pipeline (load, split, train the clean
baseline through the shared cache, adversarially train the defended model,
distil a surrogate, score fidelity, append a row) runs on CPU in seconds with no
download. What is asserted is "each cell produces a well-formed row with finite,
in-range numbers, reproducibly", never paper accuracy.

The load-bearing test here is `test_the_defended_model_is_not_the_clean_target`:
it is what proves the old `advtr_modelext.py:189` bug is not reproduced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path

# Every metric E2 records is a percentage in [0, 100].
_PERCENT_COLUMNS = (
    "target_test_acc",
    "defended_test_acc",
    "target_robust_acc",
    "defended_robust_acc",
    "stolen_test_acc",
    "fidelity",
    "correct_fidelity",
)


def _context(tmp_path: Path, seed: int = 0):
    """Build a `test`-level run context over a throwaway cache."""
    from common.config import get_level
    from experiments import shared_targets as targets

    config = get_level("test").with_defaults(epochs=100)
    torch.set_num_threads(1)
    targets.seed_everything(seed)
    return targets.RunContext(
        level=config, seed=seed, device="cpu", cache_dir=tmp_path / "models"
    )


@pytest.mark.integration
def test_test_level_run_writes_an_in_range_row(tmp_path: Path) -> None:
    """One cell produces a single CSV row whose metrics are all in [0, 100]."""
    from common.io import read_rows
    from experiments.e2_advtr_modext import run as e2

    rows = e2.run(
        level="test",
        seeds=(0,),
        datasets=("census",),
        epsilons=(0.01,),
        output_dir=tmp_path,
    )

    assert len(rows) == 1
    for column in _PERCENT_COLUMNS:
        value = cast(float, rows[0][column])
        assert 0.0 <= value <= 100.0, f"{column} outside [0, 100]: {value}"
    assert len(read_rows(tmp_path / "e2_advtr_modext.csv")) == 1


@pytest.mark.integration
def test_the_defended_model_is_not_the_clean_target(tmp_path: Path) -> None:
    """The defended model is a distinct, differently-trained network.

    This is the correctness pin for plan S5: the old script did
    `defended_model = target_model`, throwing the adversarially-trained model
    away. Here the clean and defended models are separate objects, hash to
    separate checkpoints, and hold different weights, so the "defended"
    measurements cannot secretly be the plain target's.
    """
    from experiments.e2_advtr_modext import run as e2

    ctx = _context(tmp_path)
    bundle, _ = e2.build_models(ctx, "census", 0.01)

    assert bundle.clean is not bundle.defended
    assert bundle.clean_spec.key() != bundle.defended_spec.key()

    clean_weights = list(bundle.clean.state_dict().values())
    defended_weights = list(bundle.defended.state_dict().values())
    assert any(
        not torch.equal(a, b)
        for a, b in zip(clean_weights, defended_weights, strict=True)
    ), "adversarial training left the weights identical to the clean target"


@pytest.mark.integration
def test_the_defended_metrics_are_measured_on_the_defended_model(
    tmp_path: Path,
) -> None:
    """The row's defended test accuracy is the defended model's, not the clean one's.

    A regression guard for the wiring: were the defended metric sourced from the
    clean target (the old bug's effect), this equality would read the wrong
    model whenever the two accuracies differ.
    """
    from amulet.utils import get_accuracy
    from experiments import shared_targets as targets
    from experiments.e2_advtr_modext import run as e2

    ctx = _context(tmp_path)
    bundle, data = e2.build_models(ctx, "census", 0.01)
    test_loader = targets.loader_for(data.test_set, targets.batch_for(ctx.level, 256))

    row = e2.run_cell(_context(tmp_path), "census", 0.01, tmp_path / "out")[0]

    assert cast(float, row["defended_test_acc"]) == pytest.approx(
        get_accuracy(bundle.defended, test_loader, "cpu")
    )
    assert cast(float, row["target_test_acc"]) == pytest.approx(
        get_accuracy(bundle.clean, test_loader, "cpu")
    )


@pytest.mark.integration
def test_the_clean_baseline_is_trained_once_across_budgets(tmp_path: Path) -> None:
    """Two budgets share one clean checkpoint (plan S6): 1 clean + 2 defended + 2 stolen.

    Were the clean target keyed on epsilon, this would train it twice and leave
    six checkpoints; the epsilon-independent clean spec leaves five.
    """
    from experiments.e2_advtr_modext import run as e2

    _ = e2.run(
        level="test",
        seeds=(0,),
        datasets=("census",),
        epsilons=(0.01, 0.03),
        output_dir=tmp_path,
        cache_dir=tmp_path / "models",
    )

    assert len(sorted((tmp_path / "models").glob("*.pt"))) == 5


@pytest.mark.integration
def test_the_evasion_degrades_both_models(tmp_path: Path) -> None:
    """Robust accuracy is no better than clean accuracy, for target and defended alike.

    A pipeline that evaluated the clean test set twice instead of the perturbed
    one would pass every range check; this is the cheap proof the perturbed
    loader reaches the metric.
    """
    from experiments.e2_advtr_modext import run as e2

    row = e2.run(
        level="test",
        seeds=(0,),
        datasets=("census",),
        epsilons=(0.01,),
        output_dir=tmp_path,
    )[0]

    assert cast(float, row["target_robust_acc"]) <= cast(float, row["target_test_acc"])
    assert cast(float, row["defended_robust_acc"]) <= cast(
        float, row["defended_test_acc"]
    )


@pytest.mark.integration
def test_rerunning_into_the_same_csv_does_not_duplicate(tmp_path: Path) -> None:
    """The idempotent append resumes an interrupted sweep rather than doubling it."""
    from common.io import read_rows
    from experiments.e2_advtr_modext import run as e2

    for _ in range(2):
        _ = e2.run(
            level="test",
            seeds=(0,),
            datasets=("census",),
            epsilons=(0.01,),
            output_dir=tmp_path,
            cache_dir=tmp_path / "models",
        )

    assert len(read_rows(tmp_path / "e2_advtr_modext.csv")) == 1


@pytest.mark.integration
def test_same_seed_reproduces(tmp_path: Path) -> None:
    """Two seeded runs agree numerically, so a reviewer's re-run is comparable.

    Agreement is to a tolerance, not bitwise: CPU float reductions are not
    bit-reproducible once the intra-op pool is warm, and PGD starts from a random
    point. A swapped or mis-scaled metric moves a cell far more than this.
    """
    from experiments.e2_advtr_modext import run as e2

    def once(where: Path) -> dict[str, object]:
        return e2.run(
            level="test",
            seeds=(0,),
            datasets=("census",),
            epsilons=(0.01,),
            output_dir=where / "out",
            cache_dir=where / "models",
        )[0]

    first = once(tmp_path / "a")
    second = once(tmp_path / "b")

    for column in _PERCENT_COLUMNS:
        assert first[column] == pytest.approx(second[column], abs=1.0), column


@pytest.mark.integration
def test_a_test_level_run_leaves_the_committed_results_untouched(
    tmp_path: Path,
) -> None:
    """A tiny run must never write where the paper's numbers live."""
    from common.io import results_path
    from experiments.e2_advtr_modext import run as e2

    committed = results_path("e2_advtr_modext")
    before = committed.read_text() if committed.exists() else None

    _ = e2.run(level="test", seeds=(0,), datasets=("census",), epsilons=(0.01,))

    after = committed.read_text() if committed.exists() else None
    assert after == before
