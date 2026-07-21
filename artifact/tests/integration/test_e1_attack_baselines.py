"""Tiny end-to-end run of E1, the attack baselines (plan §8, Level 1).

The `test` level substitutes a three-layer VGG for VGG11/13/16/19 and a handful
of synthetic 3x32x32 images for CelebA, so the whole pipeline — load, train the
target through the shared cache, run the attack, score it, append a CSV row —
runs on CPU in seconds with no download and no network. CelebA is a multi-
gigabyte Google Drive fetch; requiring it here would make the fast verification
tier unrunnable on a fresh clone, which defeats the purpose of Level 1.

What is asserted is "each sub-attack produces a well-formed row with finite,
in-range numbers, reproducibly", never paper accuracy. A three-layer network
trained for one epoch on 64 synthetic images has no opinion about membership
inference; that it runs end to end and writes a parseable row is the claim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from experiments.e1_attack_baselines.schemas import ATTACKS

if TYPE_CHECKING:
    from pathlib import Path

# Columns reported as percentages, which a well-formed row keeps within [0, 100].
PERCENT_COLUMNS: dict[str, tuple[str, ...]] = {
    "evasion": ("target_test_acc", "robust_acc"),
    "poisoning": (
        "std_test_acc",
        "std_poison_acc",
        "pois_test_acc",
        "pois_poison_acc",
    ),
    "model_extraction": (
        "target_test_acc",
        "stolen_test_acc",
        "fidelity",
        "correct_fidelity",
    ),
    "membership_inference": (
        "target_train_acc",
        "target_test_acc",
        "offline_bal_acc",
        "offline_tpr_at_1fpr",
        "online_bal_acc",
        "online_tpr_at_1fpr",
    ),
    "attribute_inference": ("target_test_acc", "attack_bal_acc"),
    "data_reconstruction": ("target_test_acc",),
}

# Columns reported as fractions of one: ROC AUC, and MSE over inputs in [0, 1].
UNIT_COLUMNS: dict[str, tuple[str, ...]] = {
    "evasion": (),
    "poisoning": (),
    "model_extraction": (),
    "membership_inference": ("offline_auc", "online_auc"),
    "attribute_inference": ("attack_auc",),
    "data_reconstruction": ("mse_avg", "mse_0", "mse_1"),
}


@pytest.mark.integration
@pytest.mark.parametrize("attack", ATTACKS)
def test_test_level_run_writes_an_in_range_row(tmp_path: Path, attack: str) -> None:
    """Each sub-attack produces one CSV row whose metrics are in range."""
    from common.io import read_rows
    from experiments.e1_attack_baselines import run as e1

    rows = e1.run(
        level="test",
        seeds=(0,),
        attacks=(attack,),
        capacities=("m1",),
        output_dir=tmp_path,
    )

    assert len(rows) == 1
    for column in PERCENT_COLUMNS[attack]:
        value = cast(float, rows[0][column])
        assert 0.0 <= value <= 100.0, f"{column} outside [0, 100]: {value}"
    for column in UNIT_COLUMNS[attack]:
        value = cast(float, rows[0][column])
        assert 0.0 <= value <= 1.0, f"{column} outside [0, 1]: {value}"

    written = read_rows(tmp_path / f"{attack}.csv")
    assert len(written) == 1


@pytest.mark.integration
@pytest.mark.parametrize("attack", ATTACKS)
def test_same_seed_reproduces(tmp_path: Path, attack: str) -> None:
    """Two runs of one seed agree numerically, so a reviewer's re-run is comparable.

    Reproducibility is the property the artifact is judged on, and the one the
    old scripts did not have: three of them seeded only `torch`, leaving NumPy's
    global generator to decide which records the adversary saw. Seeding every
    generator (`shared.seed_everything`) is what makes this hold.

    Agreement is asserted to a numerical tolerance, not bitwise. On CPU a warm
    intra-op thread pool makes float reductions non-bit-reproducible regardless
    of seeding (the training here agrees only after forcing a single thread, and
    even then the pool may already be warm under pytest). The plan anticipates
    exactly this: retraining legitimately yields numerically-close numbers, and
    a bitwise assertion would flake on thread scheduling rather than catch a real
    regression. A swapped or mis-scaled metric moves a cell by far more than the
    tolerance here.
    """
    from experiments.e1_attack_baselines import run as e1

    def once(where: Path) -> dict[str, object]:
        return e1.run(
            level="test",
            seeds=(0,),
            attacks=(attack,),
            capacities=("m1",),
            output_dir=where,
        )[0]

    first = once(tmp_path / "a")
    second = once(tmp_path / "b")

    for column in PERCENT_COLUMNS[attack]:
        assert first[column] == pytest.approx(second[column], abs=0.5), column
    for column in UNIT_COLUMNS[attack]:
        assert first[column] == pytest.approx(second[column], abs=0.02), column


@pytest.mark.integration
def test_rerunning_into_the_same_csv_does_not_duplicate(tmp_path: Path) -> None:
    """The idempotent append means an interrupted sweep is resumed, not doubled."""
    from common.io import read_rows
    from experiments.e1_attack_baselines import run as e1

    for _ in range(2):
        _ = e1.run(
            level="test",
            seeds=(0,),
            attacks=("evasion",),
            capacities=("m1",),
            output_dir=tmp_path,
        )

    assert len(read_rows(tmp_path / "evasion.csv")) == 1


@pytest.mark.integration
def test_the_evasion_attack_actually_degrades_the_target(tmp_path: Path) -> None:
    """Adversarial examples score no better than the clean inputs they perturb.

    A pipeline that silently evaluated the clean test set twice would produce
    two in-range numbers and pass every other assertion here. This is the cheap
    sanity check that the perturbed loader reaches the metric.
    """
    from experiments.e1_attack_baselines import run as e1

    row = e1.run(
        level="test",
        seeds=(0,),
        attacks=("evasion",),
        capacities=("m1",),
        output_dir=tmp_path,
    )[0]

    assert cast(float, row["robust_acc"]) <= cast(float, row["target_test_acc"])


@pytest.mark.integration
def test_extraction_and_attribute_inference_train_one_target_between_them(
    tmp_path: Path,
) -> None:
    """The shared target is trained once and reloaded, not trained twice.

    This is the payoff of the content-addressed cache (plan §6): the two
    adversary-split attacks agree on a recipe, so the second one to run finds
    the first one's checkpoint. On the paper's settings that is a 100-epoch VGG
    training saved per capacity per seed.
    """
    from experiments.e1_attack_baselines import run as e1

    _ = e1.run(
        level="test",
        seeds=(0,),
        attacks=("model_extraction", "attribute_inference"),
        capacities=("m1",),
        output_dir=tmp_path,
        cache_dir=tmp_path / "models",
    )

    # One shared target plus the stolen surrogate only extraction trains.
    assert len(sorted((tmp_path / "models").glob("*.pt"))) == 2


@pytest.mark.integration
def test_a_test_level_run_leaves_the_committed_results_untouched(
    tmp_path: Path,
) -> None:
    """A tiny run must never write where the paper's numbers live.

    A three-layer network's accuracy averaged into the shipped CSVs would
    silently corrupt the table, so `test` defaults to a throwaway directory.
    """
    from common.io import results_path
    from experiments.e1_attack_baselines import run as e1

    committed = results_path("e1_attack_baselines", "evasion")
    before = committed.read_text() if committed.exists() else None

    _ = e1.run(level="test", seeds=(0,), attacks=("evasion",), capacities=("m1",))

    after = committed.read_text() if committed.exists() else None
    assert after == before
