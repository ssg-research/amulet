"""Contract for the shared adversarial-training model specs (plan S5, S6, P3).

E2 and E3 share one defense, so they share `advtr_common`'s spec builders. These
pure tests pin the sharing rules the content-addressed cache depends on:

* the clean baseline $\\modelstd$ is epsilon-independent, so one is trained per
  (dataset, seed) and reused across every budget;
* each adversarially-trained $\\modeldef$ is its own checkpoint per budget;
* $\\modelstd$ and $\\modeldef$ are *different* checkpoints, which is the bug the
  old `advtr_modelext.py:189` introduced and this port refuses to reproduce;
* E2 and E3 split the training data differently, so their same-dataset targets
  never collide on one key;
* a tiny `test`-level stand-in can never reuse a paper checkpoint.

No data, no training, no GPU: these are functions of the spec builders alone.
"""

from __future__ import annotations

import pytest

from common.config import LevelConfig, get_level
from common.models import ModelSpec
from experiments import advtr_common as advtr
from experiments.e2_advtr_modext.schemas import EPSILONS as E2_EPSILONS
from experiments.e3_advtr_attrinf.schemas import EPSILONS as E3_EPSILONS

# The paper full-level budget: one seed, whole split, 100 epochs.
LEVEL = get_level("full").with_defaults(epochs=100)
TINY = get_level("test").with_defaults(epochs=100)

# Representative shapes; the exact values only need to be internally consistent.
NUM_FEATURES = 93
NUM_CLASSES = 2
CAPACITY = "m1"
BATCH = 256


def _clean(
    dataset: str, selector: str, seed: int = 0, level: LevelConfig = LEVEL
) -> ModelSpec:
    return advtr.clean_target_spec(
        level, dataset, seed, CAPACITY, NUM_FEATURES, NUM_CLASSES, BATCH, selector
    )


def _defended(
    dataset: str,
    selector: str,
    epsilon: float,
    seed: int = 0,
    level: LevelConfig = LEVEL,
) -> ModelSpec:
    return advtr.defended_target_spec(
        level,
        dataset,
        seed,
        CAPACITY,
        NUM_FEATURES,
        NUM_CLASSES,
        BATCH,
        epsilon,
        selector,
    )


def test_the_clean_baseline_is_epsilon_independent() -> None:
    """One $\\modelstd$ serves every budget: its spec does not depend on epsilon.

    This is what lets the sweep train the clean target once per (dataset, seed)
    and reuse it across all four budgets, saving three full trainings per cell.
    """
    first = _clean("census", advtr.DATASET_SPLIT_TARGET)
    second = _clean("census", advtr.DATASET_SPLIT_TARGET)

    assert first == second
    assert first.key() == second.key()


def test_each_budget_is_its_own_defended_checkpoint() -> None:
    """Two $\\modeldef$ specs differing only in epsilon get different keys."""
    keys = {
        _defended("census", advtr.DATASET_SPLIT_TARGET, eps).key()
        for eps in E2_EPSILONS
    }

    assert len(keys) == len(E2_EPSILONS)


def test_the_defended_model_is_not_the_clean_target() -> None:
    """$\\modelstd$ and $\\modeldef$ hash apart, so neither can load the other.

    The old `advtr_modelext.py:189` did `defended_model = target_model`,
    collapsing the two. Recording the optimizer recipe (Adam vs adversarial PGD)
    in the spec is what makes that collapse impossible here.
    """
    clean = _clean("census", advtr.DATASET_SPLIT_TARGET)
    for epsilon in E2_EPSILONS:
        defended = _defended("census", advtr.DATASET_SPLIT_TARGET, epsilon)
        assert clean.key() != defended.key()
        assert clean.optimizer_recipe != defended.optimizer_recipe


def test_e2_and_e3_targets_do_not_collide_on_one_key() -> None:
    """A census target split for E2 is a different checkpoint from E3's.

    E2 splits the adversary's half at the dataset level; E3 splits the NumPy
    arrays by index. The two halves differ, so the subset selectors differ, so
    the keys differ: E3 can never load an E2 census target in its place.
    """
    e2 = _clean("census", advtr.DATASET_SPLIT_TARGET)
    e3 = _clean("census", advtr.ARRAY_SPLIT_TARGET)

    assert e2.subset_selector != e3.subset_selector
    assert e2.key() != e3.key()


def test_the_stolen_surrogate_is_its_own_checkpoint() -> None:
    """$\\modelstol$ hashes apart from the clean target and every defended one.

    It trains on the adversary's half with a distillation recipe naming its
    source budget, so no clean target, defended target, or surrogate of a
    different budget can be reused in its place.
    """
    clean = _clean("cifar", advtr.DATASET_SPLIT_TARGET)
    stolen = {
        advtr.stolen_model_spec(
            LEVEL, "cifar", 0, CAPACITY, NUM_FEATURES, NUM_CLASSES, BATCH, eps
        ).key()
        for eps in E2_EPSILONS
    }
    defended = {
        _defended("cifar", advtr.DATASET_SPLIT_TARGET, eps).key() for eps in E2_EPSILONS
    }

    assert len(stolen) == len(E2_EPSILONS)
    assert clean.key() not in stolen
    assert stolen.isdisjoint(defended)


@pytest.mark.parametrize(
    ("dataset", "expected_arch"),
    [
        ("census", "linearnet"),
        ("lfw", "linearnet"),
        ("fmnist", "cnn"),
        ("cifar", "vgg"),
    ],
)
def test_each_dataset_trains_its_modality_appropriate_architecture(
    dataset: str, expected_arch: str
) -> None:
    """Tabular datasets get a dense net, single-channel fmnist a CNN, cifar a VGG.

    VGG cannot run on the tabular (census, lfw) or single-channel (fmnist)
    inputs, so the per-modality choice is what the library actually admits, and
    it is recorded in the spec so a cross-modality mix-up is a different key.
    """
    spec = _clean(dataset, advtr.DATASET_SPLIT_TARGET)

    assert spec.arch == expected_arch


def test_the_tiny_test_level_target_cannot_reuse_a_paper_checkpoint() -> None:
    """A `test`-level run records a distinct architecture, so it cannot collide."""
    paper = _clean("census", advtr.DATASET_SPLIT_TARGET)
    tiny = _clean("census", advtr.DATASET_SPLIT_TARGET, level=TINY)

    assert tiny.arch != paper.arch
    assert tiny.key() != paper.key()


@pytest.mark.parametrize("epsilon", E3_EPSILONS)
def test_a_different_seed_is_a_different_defended_model(epsilon: float) -> None:
    """Sharing must not leak across seeds: seed 0 and seed 1 differ in key."""
    first = _defended("lfw", advtr.ARRAY_SPLIT_TARGET, epsilon, seed=0)
    second = _defended("lfw", advtr.ARRAY_SPLIT_TARGET, epsilon, seed=1)

    assert first.key() != second.key()
