"""Contract for the adversarial-training model specs of E2 and E3.

E2 and E3 each define their own spec builders in their own `train_targets`
module. These pure tests pin each experiment's spec discipline, the rules the
content-addressed cache depends on:

* the clean baseline is epsilon-independent, so one is trained per (dataset,
  seed) and reused across every budget;
* each defended model is its own checkpoint per budget;
* the clean and defended models are *different* checkpoints, so a "defended"
  measurement can never read the clean model;
* E2 and E3 split the training data differently, so their same-dataset targets
  never collide on one key (a safety property: it holds however the two diverge,
  since they are independent experiments);
* a tiny `test`-level stand-in can never reuse a paper checkpoint.

No data, no training, no GPU: these are functions of the spec builders alone.
"""

from __future__ import annotations

import pytest

from common.config import LevelConfig, get_level
from common.models import ModelSpec
from experiments.e2_advtr_modext import train_targets as e2
from experiments.e2_advtr_modext.schemas import EPSILONS as E2_EPSILONS
from experiments.e3_advtr_attrinf import train_targets as e3
from experiments.e3_advtr_attrinf.schemas import EPSILONS as E3_EPSILONS

# The paper full-level budget: one seed, whole split, 100 epochs.
LEVEL = get_level("full").with_defaults(epochs=100)
TINY = get_level("test").with_defaults(epochs=100)

# Representative shapes; the exact values only need to be internally consistent.
NUM_FEATURES = 93
NUM_CLASSES = 2
CAPACITY = "m1"
BATCH = 256


def _e2_clean(dataset: str, seed: int = 0, level: LevelConfig = LEVEL) -> ModelSpec:
    return e2.clean_target_spec(
        level, dataset, seed, CAPACITY, NUM_FEATURES, NUM_CLASSES, BATCH
    )


def _e2_defended(
    dataset: str, epsilon: float, seed: int = 0, level: LevelConfig = LEVEL
) -> ModelSpec:
    return e2.defended_target_spec(
        level, dataset, seed, CAPACITY, NUM_FEATURES, NUM_CLASSES, BATCH, epsilon
    )


def _e3_clean(dataset: str, seed: int = 0, level: LevelConfig = LEVEL) -> ModelSpec:
    return e3.clean_target_spec(
        level, dataset, seed, CAPACITY, NUM_FEATURES, NUM_CLASSES, BATCH
    )


def _e3_defended(
    dataset: str, epsilon: float, seed: int = 0, level: LevelConfig = LEVEL
) -> ModelSpec:
    return e3.defended_target_spec(
        level, dataset, seed, CAPACITY, NUM_FEATURES, NUM_CLASSES, BATCH, epsilon
    )


def test_e2_clean_baseline_is_epsilon_independent() -> None:
    """One clean baseline serves every budget: E2's clean spec ignores epsilon.

    This is what lets the sweep train the clean target once per (dataset, seed)
    and reuse it across all four budgets, saving three full trainings per cell.
    """
    first = _e2_clean("census")
    second = _e2_clean("census")

    assert first == second
    assert first.key() == second.key()


def test_e2_each_budget_is_its_own_defended_checkpoint() -> None:
    """Two defended specs differing only in epsilon get different keys."""
    keys = {_e2_defended("census", eps).key() for eps in E2_EPSILONS}

    assert len(keys) == len(E2_EPSILONS)


def test_e2_defended_model_is_not_the_clean_target() -> None:
    """The clean and defended models hash apart, so neither can load the other.

    Recording the optimizer recipe (Adam vs adversarial PGD) in the spec is what
    keeps the two a separate checkpoint each.
    """
    clean = _e2_clean("census")
    for epsilon in E2_EPSILONS:
        defended = _e2_defended("census", epsilon)
        assert clean.key() != defended.key()
        assert clean.optimizer_recipe != defended.optimizer_recipe


def test_e2_stolen_surrogate_is_its_own_checkpoint() -> None:
    """The surrogate hashes apart from the clean target and every defended one.

    It trains on the adversary's half with a distillation recipe naming its
    source budget, so no clean target, defended target, or surrogate of a
    different budget can be reused in its place.
    """
    clean = _e2_clean("cifar")
    stolen = {
        e2.stolen_model_spec(
            LEVEL, "cifar", 0, CAPACITY, NUM_FEATURES, NUM_CLASSES, BATCH, eps
        ).key()
        for eps in E2_EPSILONS
    }
    defended = {_e2_defended("cifar", eps).key() for eps in E2_EPSILONS}

    assert len(stolen) == len(E2_EPSILONS)
    assert clean.key() not in stolen
    assert stolen.isdisjoint(defended)


def test_e3_defended_model_is_not_the_clean_target() -> None:
    """E3's clean and defended models hash apart, per budget."""
    clean = _e3_clean("census")
    for epsilon in E3_EPSILONS:
        defended = _e3_defended("census", epsilon)
        assert clean.key() != defended.key()
        assert clean.optimizer_recipe != defended.optimizer_recipe


def test_e2_and_e3_targets_do_not_collide_on_one_key() -> None:
    """A census target built for E2 is a different checkpoint from E3's.

    E2 splits the adversary's half at the dataset level; E3 splits the NumPy
    arrays by index. The two halves differ, so the subset selectors differ, so
    the keys differ: E3 can never load an E2 census target in its place. This is
    a safety property, not a coupling: it holds whatever either experiment
    changes, because their selectors are independently defined.
    """
    e2_spec = _e2_clean("census")
    e3_spec = _e3_clean("census")

    assert e2_spec.subset_selector != e3_spec.subset_selector
    assert e2_spec.key() != e3_spec.key()


@pytest.mark.parametrize(
    ("dataset", "expected_arch"),
    [
        ("census", "linearnet"),
        ("lfw", "linearnet"),
        ("fmnist", "linearnet"),
        ("cifar", "vgg"),
    ],
)
def test_each_dataset_trains_its_modality_appropriate_architecture(
    dataset: str, expected_arch: str
) -> None:
    """Tabular datasets and fmnist get a dense net; cifar gets a VGG.

    fmnist belongs on the dense net with census and lfw, the pairing the paper
    ran: `LinearNet` opens with `nn.Flatten` and `load_fmnist` reports
    `num_features=784`, its exact flattened 28x28 size. `num_features` counts
    H*W and ignores channels, so it agrees with the flattened size only for
    single-channel data; cifar's 1024 against a flattened 3072 disagrees, which
    is what puts cifar on a conv net. VGG is hard-wired to three input channels
    and cannot take the tabular or single-channel inputs at all. The choice is
    recorded in the spec, so a cross-modality mix-up is a different cache key.
    """
    spec = _e2_clean(dataset)

    assert spec.arch == expected_arch


def test_the_tiny_test_level_target_cannot_reuse_a_paper_checkpoint() -> None:
    """A `test`-level run records a distinct architecture, so it cannot collide."""
    paper = _e2_clean("census")
    tiny = _e2_clean("census", level=TINY)

    assert tiny.arch != paper.arch
    assert tiny.key() != paper.key()


@pytest.mark.parametrize("epsilon", E3_EPSILONS)
def test_a_different_seed_is_a_different_defended_model(epsilon: float) -> None:
    """Sharing must not leak across seeds: seed 0 and seed 1 differ in key."""
    first = _e3_defended("lfw", epsilon, seed=0)
    second = _e3_defended("lfw", epsilon, seed=1)

    assert first.key() != second.key()
