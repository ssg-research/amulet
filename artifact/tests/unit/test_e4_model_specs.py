"""Contract for E4's model specs and the E2<->E4 baseline decision (plan S6, S13, P4).

E4 has no original script; it composes the outlier removal defense with model
extraction. Its clean baseline is a clean model-extraction target on the same
four datasets as E2, so the plan asks a deliberate question: does E4's baseline
*share* E2's checkpoint, or is it a separate one?

**Decision: shared.** Both experiments describe the clean target with the
identical spec (the 50/50 dataset-level split selector `DATASET_SPLIT_TARGET`,
the Adam recipe, 100 epochs, batch 256), so their content hashes coincide and
the cache stores one $\\modelstd$ serving both. These pure tests prove it, and
prove the flip side: every outlier-removed model is its own checkpoint, distinct
from the baseline, from every other removal percentage, and from any E2 defended
model (which encodes an epsilon where E4 encodes a percentage).

No data, no training, no GPU: functions of the spec builders alone.
"""

from __future__ import annotations

import pytest

from common.config import LevelConfig, get_level
from experiments import advtr_common as advtr
from experiments.e2_advtr_modext import run as e2
from experiments.e2_advtr_modext.schemas import EPSILONS as E2_EPSILONS
from experiments.e4_outrem_modext import run as e4
from experiments.e4_outrem_modext.schemas import PERCENTS

# The paper full-level budget: one seed, whole split, 100 epochs.
LEVEL = get_level("full").with_defaults(epochs=100)
TINY = get_level("test").with_defaults(epochs=100)

# Representative shapes; the exact values only need to be internally consistent.
NUM_FEATURES = 93
NUM_CLASSES = 2
CAPACITY = "m1"


def _e4_context(level: LevelConfig = LEVEL, seed: int = 0) -> advtr.RunContext:
    return advtr.RunContext(level=level, seed=seed, device="cpu")


def _e4_clean(dataset: str, level: LevelConfig = LEVEL, seed: int = 0):
    return e4.clean_baseline_spec(
        _e4_context(level, seed), dataset, NUM_FEATURES, NUM_CLASSES, e4.BATCH_SIZE
    )


def _e2_clean(dataset: str, level: LevelConfig = LEVEL, seed: int = 0):
    # Built exactly as E2's `build_models` builds its clean baseline.
    return advtr.clean_target_spec(
        level,
        dataset,
        seed,
        CAPACITY,
        NUM_FEATURES,
        NUM_CLASSES,
        e2.BATCH_SIZE,
        advtr.DATASET_SPLIT_TARGET,
    )


def _e4_defended(dataset: str, percent: int, level: LevelConfig = LEVEL, seed: int = 0):
    # Built exactly as E4's `build_models` builds a removed model.
    return _e4_clean(dataset, level, seed).replace(
        optimizer_recipe=e4.outrem_recipe(percent)
    )


def _e4_stolen(dataset: str, percent: int, level: LevelConfig = LEVEL, seed: int = 0):
    return _e4_clean(dataset, level, seed).replace(
        optimizer_recipe=e4.stolen_recipe(percent),
        subset_selector=advtr.DATASET_SPLIT_ADVERSARY,
    )


@pytest.mark.parametrize("dataset", ("census", "lfw", "fmnist", "cifar"))
def test_e4_baseline_shares_e2s_clean_target_checkpoint(dataset: str) -> None:
    """E4's clean baseline hashes identically to E2's, so one checkpoint serves both.

    This is the E2<->E4 sharing decision (plan S6, S13) made concrete: same
    dataset, arch, split selector, recipe, epochs and batch means the same
    content hash. If either experiment changed any weight-affecting field, this
    equality would break and they would (correctly) stop sharing.
    """
    e2_spec = _e2_clean(dataset)
    e4_spec = _e4_clean(dataset)

    assert e2_spec == e4_spec
    assert e2_spec.key() == e4_spec.key()


def test_the_shared_recipe_and_selector_are_the_reference_ones() -> None:
    """The baseline uses the P3-recorded selector and recipe the sharing depends on."""
    spec = _e4_clean("census")

    assert spec.subset_selector == advtr.DATASET_SPLIT_TARGET
    assert spec.optimizer_recipe == advtr.ADAM_RECIPE
    assert spec.epochs == 100
    assert e4.BATCH_SIZE == e2.BATCH_SIZE == 256


def test_each_removal_percentage_is_its_own_defended_checkpoint() -> None:
    """Two defended specs differing only in the removal percentage get different keys.

    The nonzero percentages each hash apart; `0` is not built through the removal
    recipe at all (it is the shared clean baseline), so it is excluded here.
    """
    nonzero = [p for p in PERCENTS if p != 0]
    keys = {_e4_defended("census", p).key() for p in nonzero}

    assert len(keys) == len(nonzero)


def test_a_removed_model_is_not_the_clean_baseline() -> None:
    """Every outlier-removed $\\modeldef$ hashes apart from the clean baseline.

    Recording the removal percentage in the optimizer recipe is what makes a
    removed model a distinct checkpoint rather than a reload of $\\modelstd$.
    """
    clean = _e4_clean("census")
    for percent in PERCENTS:
        if percent == 0:
            continue
        defended = _e4_defended("census", percent)
        assert clean.key() != defended.key()
        assert clean.optimizer_recipe != defended.optimizer_recipe


def test_e4_removed_models_never_collide_with_e2_defended_models() -> None:
    """An outlier-removed model and an adversarially-trained one cannot share a key.

    E4 encodes a removal percentage in the recipe; E2 encodes an epsilon. The two
    recipe namespaces are disjoint, so no cross-experiment defended checkpoint is
    ever reused in the wrong place.
    """
    e4_keys = {_e4_defended("cifar", p).key() for p in PERCENTS if p != 0}
    e2_keys = {
        advtr.defended_target_spec(
            LEVEL,
            "cifar",
            0,
            CAPACITY,
            NUM_FEATURES,
            NUM_CLASSES,
            e2.BATCH_SIZE,
            eps,
            advtr.DATASET_SPLIT_TARGET,
        ).key()
        for eps in E2_EPSILONS
    }

    assert e4_keys.isdisjoint(e2_keys)


def test_the_stolen_surrogate_is_its_own_checkpoint() -> None:
    """$\\modelstol$ hashes apart from the clean target and every removed model.

    It trains on the adversary's half with a distillation recipe naming its
    source removal percentage, so no clean target, removed target, or surrogate
    of a different percentage can be reused in its place.
    """
    clean = _e4_clean("cifar")
    stolen = {_e4_stolen("cifar", p).key() for p in PERCENTS}
    removed = {_e4_defended("cifar", p).key() for p in PERCENTS if p != 0}

    assert len(stolen) == len(PERCENTS)
    assert clean.key() not in stolen
    assert stolen.isdisjoint(removed)


def test_the_tiny_test_level_baseline_cannot_reuse_a_paper_checkpoint() -> None:
    """A `test`-level run records a distinct architecture, so it cannot collide."""
    paper = _e4_clean("census")
    tiny = _e4_clean("census", level=TINY)

    assert tiny.arch != paper.arch
    assert tiny.key() != paper.key()
