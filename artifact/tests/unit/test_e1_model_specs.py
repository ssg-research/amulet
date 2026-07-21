"""Contract for E1's target-model sharing (plan §6, §7.1, §12 P2).

The point of routing every E1 model through `common.models.ModelSpec` is that
sharing becomes a property of the *recipe* rather than of which script ran
first: two sub-attacks that need the identical model get one checkpoint
automatically, and two that need different models cannot collide however similar
their code looks.

E1 is the hard case for that claim, because its six sub-attacks deliberately
diverge along three independent axes (plan §5):

* optimizer recipe — evasion and poisoning train with SGD + StepLR, everything
  else with Adam;
* label attribute — membership inference and data reconstruction predict
  CelebA's `Wavy_Hair`, the rest `Smiling`;
* training subset — model extraction and attribute inference reserve half the
  training split for the adversary, data reconstruction uses all of it.

These tests are pure functions of the spec builders: no data, no training, no
GPU. They are what makes an accidental future edit that collapses two recipes
into one (or silently splits a shared one) fail fast.
"""

from __future__ import annotations

import pytest

from common.config import get_level
from common.models import ModelSpec
from experiments.e1_attack_baselines import shared

# The paper's full-level budget: 100 epochs, one seed, the whole training split.
LEVEL = get_level("full").with_defaults(epochs=shared.PAPER_EPOCHS)

# CelebA as `amulet.utils.load_data` reports it: 64x64 images, binary target.
NUM_FEATURES = 64 * 64
NUM_CLASSES = 2


def _spec(builder_name: str, seed: int = 0, capacity: str = "m1") -> ModelSpec:
    """Build one named target spec at the paper's settings."""
    builder = getattr(shared, builder_name)
    return builder(LEVEL, seed, capacity, NUM_FEATURES, NUM_CLASSES)


# Every distinct model E1 trains, by the spec builder that describes it.
TARGET_BUILDERS = (
    "evasion_target_spec",
    "poisoning_clean_spec",
    "poisoning_backdoored_spec",
    "adversary_split_target_spec",
    "stolen_model_spec",
    "reconstruction_target_spec",
    "overfit_target_spec",
)


def test_every_e1_model_recipe_has_its_own_cache_key() -> None:
    """No two of E1's seven distinct models collide on one checkpoint.

    A collision would mean one sub-attack silently loading another's weights,
    which is the exact failure the content-addressed cache exists to prevent.
    """
    keys = [_spec(name).key() for name in TARGET_BUILDERS]

    assert len(set(keys)) == len(TARGET_BUILDERS)


def test_model_extraction_and_attribute_inference_share_one_target() -> None:
    """The two adversary-split attacks train one target between them.

    Both reserve 50% of the training split for the adversary and train the
    target on the other half with Adam against the `Smiling` label. That is the
    same model, so it must be trained once and reused, saving a full 100-epoch
    VGG training per capacity per seed.
    """
    from experiments.e1_attack_baselines import attribute_inference, model_extraction

    extraction = model_extraction.target_spec(LEVEL, 0, "m1", NUM_FEATURES, NUM_CLASSES)
    inference = attribute_inference.target_spec(
        LEVEL, 0, "m1", NUM_FEATURES, NUM_CLASSES
    )

    assert extraction == inference


@pytest.mark.parametrize(
    ("first", "second", "field"),
    [
        # SGD + StepLR versus Adam: the recipe divergence of plan §5.
        ("evasion_target_spec", "adversary_split_target_spec", "optimizer_recipe"),
        # `Smiling` versus `Wavy_Hair`: a different label is a different model.
        (
            "adversary_split_target_spec",
            "reconstruction_target_spec",
            "label_attribute",
        ),
        # The adversary's half is withheld from the target; reconstruction sees all.
        (
            "adversary_split_target_spec",
            "reconstruction_target_spec",
            "subset_selector",
        ),
    ],
)
def test_diverging_recipes_differ_in_the_field_that_explains_why(
    first: str, second: str, field: str
) -> None:
    """Each deliberate divergence is visible in the spec, not just in the hash.

    The sidecar written next to every checkpoint records these fields verbatim,
    so a reviewer can see *why* two models were not shared without recomputing
    anything.
    """
    assert getattr(_spec(first), field) != getattr(_spec(second), field)


def test_the_membership_inference_target_is_never_a_vgg() -> None:
    """The overfit target is a ResNet on a tenth of the data, as the paper says.

    Plan §5 marks this an intentional special case that must never share a
    target with anything else. Recording the architecture and the reduced
    training fraction in the spec is what enforces that.
    """
    spec = _spec("overfit_target_spec")

    assert spec.arch == "resnet"
    assert spec.train_fraction == pytest.approx(shared.OVERFIT_TRAINING_SIZE)


@pytest.mark.parametrize("builder", TARGET_BUILDERS)
@pytest.mark.parametrize(("seed", "capacity"), [(1, "m1"), (0, "m2")])
def test_a_different_seed_or_capacity_is_a_different_model(
    builder: str, seed: int, capacity: str
) -> None:
    """Sharing must not leak across the two axes E1 sweeps.

    Membership inference is reported for one capacity only, but its spec still
    varies with capacity so that a future column cannot silently reuse `m1`.
    """
    assert _spec(builder).key() != _spec(builder, seed, capacity).key()


def test_the_tiny_test_level_model_cannot_reuse_a_paper_checkpoint() -> None:
    """A `test`-level run records a different architecture, so it cannot collide.

    The tiny stand-in shares the sub-attack's optimizer recipe and subset, and
    without this distinction it would hash to the same key as the real VGG and
    be loaded in its place on the next full run.
    """
    tiny = get_level("test").with_defaults(epochs=shared.PAPER_EPOCHS)

    paper_spec = _spec("evasion_target_spec")
    tiny_spec = shared.evasion_target_spec(tiny, 0, "m1", NUM_FEATURES, NUM_CLASSES)

    assert tiny_spec.arch != paper_spec.arch
    assert tiny_spec.key() != paper_spec.key()
