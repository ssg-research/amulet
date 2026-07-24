"""Contract for E4's outlier-removal model specs.

E4 has no original script; it composes the outlier removal defense with model
extraction. Its clean baseline is a clean model-extraction target on the same
four datasets as E2, built through the shared `clean_target_spec` with the 50/50
dataset-level split selector. Whether that baseline happens to land on the same
cached checkpoint as E2's is an emergent property of the content hash, not a
contract these tests enforce: E2 and E4 are independent experiments, and each is
free to change a weight-affecting field without the other caring.

What these pure tests do pin is E4's own spec discipline: every outlier-removed
model is its own checkpoint, distinct from the baseline and from every other
removal percentage; the surrogate is distinct from both; and the removed-model
recipe namespace stays disjoint from the adversarial-training recipe namespace,
so an outlier-removed checkpoint can never be reused where a defended one is
wanted.

No data, no training, no GPU: functions of the spec builders alone.
"""

from __future__ import annotations

from common.config import LevelConfig, get_level
from common.run_context import RunContext, default_cache_dir
from experiments.e2_advtr_modext import train_targets as e2
from experiments.e2_advtr_modext.schemas import EPSILONS as E2_EPSILONS
from experiments.e4_outrem_modext import train_targets as e4
from experiments.e4_outrem_modext.schemas import PERCENTS

# The paper full-level budget: one seed, whole split, 100 epochs.
LEVEL = get_level("full").with_defaults(epochs=100)
TINY = get_level("test").with_defaults(epochs=100)

# Representative shapes; the exact values only need to be internally consistent.
NUM_FEATURES = 93
NUM_CLASSES = 2
CAPACITY = "m1"

# The batch size an adversarial-training defended spec is built at, for the
# recipe-namespace disjointness check. Not tied to E4's own batch: the point is
# only that the two recipe families never collide.
ADVTR_BATCH_SIZE = 256


def _e4_context(level: LevelConfig = LEVEL, seed: int = 0) -> RunContext:
    # These tests only build specs, so the cache is named but never written to.
    return RunContext(
        level=level, seed=seed, device="cpu", cache_dir=default_cache_dir(level)
    )


def _e4_clean(dataset: str, level: LevelConfig = LEVEL, seed: int = 0):
    return e4.clean_baseline_spec(
        _e4_context(level, seed), dataset, NUM_FEATURES, NUM_CLASSES, e4.BATCH_SIZE
    )


def _e4_defended(dataset: str, percent: int, level: LevelConfig = LEVEL, seed: int = 0):
    # Built exactly as E4's `build_models` builds a removed model.
    return _e4_clean(dataset, level, seed).replace(
        optimizer_recipe=e4.outrem_recipe(percent)
    )


def _e4_stolen(dataset: str, percent: int, level: LevelConfig = LEVEL, seed: int = 0):
    return _e4_clean(dataset, level, seed).replace(
        optimizer_recipe=e4.stolen_recipe(percent),
        subset_selector=e4.DATASET_SPLIT_ADVERSARY,
    )


def test_the_baseline_uses_the_reference_selector_and_recipe() -> None:
    """E4's clean baseline uses the reference selector, recipe, epochs and batch."""
    spec = _e4_clean("census")

    assert spec.subset_selector == e4.DATASET_SPLIT_TARGET
    assert spec.optimizer_recipe == e4.ADAM_RECIPE
    assert spec.epochs == 100
    assert e4.BATCH_SIZE == 256


def test_each_removal_percentage_is_its_own_defended_checkpoint() -> None:
    """Two defended specs differing only in the removal percentage get different keys.

    The nonzero percentages each hash apart; `0` is not built through the removal
    recipe at all (it is the shared clean baseline), so it is excluded here.
    """
    nonzero = [p for p in PERCENTS if p != 0]
    keys = {_e4_defended("census", p).key() for p in nonzero}

    assert len(keys) == len(nonzero)


def test_a_removed_model_is_not_the_clean_baseline() -> None:
    """Every outlier-removed defended model hashes apart from the clean baseline.

    Recording the removal percentage in the optimizer recipe is what makes a
    removed model a distinct checkpoint rather than a reload of the clean baseline.
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
        e2.defended_target_spec(
            LEVEL,
            "cifar",
            0,
            CAPACITY,
            NUM_FEATURES,
            NUM_CLASSES,
            ADVTR_BATCH_SIZE,
            eps,
        ).key()
        for eps in E2_EPSILONS
    }

    assert e4_keys.isdisjoint(e2_keys)


def test_the_stolen_surrogate_is_its_own_checkpoint() -> None:
    """The surrogate hashes apart from the clean target and every removed model.

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
