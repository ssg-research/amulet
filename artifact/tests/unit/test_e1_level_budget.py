"""E1's level budget: only `full` pays the paper's per-cell costs.

Two knobs dominate E1's wall clock, and both are quantities rather than code
paths: LiRA's shadow bank and data reconstruction's inversion budget. Raising
either sharpens an estimate without executing a line the smaller value did not,
so a reduced-budget level has nothing to gain from the paper's setting and hours
to lose. Both were previously reduced only at `test` level, which left `smoke`
paying full price.

Every shadow model is a full training run, and the bank is the single largest
cost in E1. The models share one architecture and one dataset, so a bigger bank
covers no extra code; it only sharpens the per-example Gaussian fit LiRA scores
with, which is attack quality that no reduced-budget level claims to measure.

The bank cannot shrink without limit. `prepare_shadow_models` ranks each
example's random draws across the bank, so every example is IN exactly
`int(PKEEP * n)` shadows; below two a side the fit has no spread to estimate,
and `LiRA.__lira_online` truncates every example's IN-set to the shortest one,
so a single example IN none of them would empty the array and turn every score
into a NaN. These tests pin both ends: the reduction happens, and it stays above
that floor.

No data, no model, no GPU: functions of the level preset alone.
"""

from __future__ import annotations

import pytest

from common.config import LEVEL_NAMES, get_level
from experiments.e1_attack_baselines import data_reconstruction, shared


def _context(name: str) -> shared.RunContext:
    """A context at one level; these tests read its knobs and never train."""
    level = get_level(name).with_defaults(epochs=1)
    return shared.RunContext(
        level=level, seed=0, device="cpu", cache_dir=shared.default_cache_dir(level)
    )


def test_full_trains_the_papers_shadow_bank() -> None:
    """`full` is the reproduction; its bank is the published one."""
    assert shared.shadow_count(get_level("full")) == shared.NUM_SHADOW
    assert shared.NUM_SHADOW == 64


@pytest.mark.parametrize("name", ("test", "smoke"))
def test_reduced_levels_train_a_smaller_bank(name: str) -> None:
    """A cheap level must not silently inherit 64 model trainings."""
    assert shared.shadow_count(get_level(name)) < shared.NUM_SHADOW


@pytest.mark.parametrize("name", LEVEL_NAMES)
def test_every_bank_leaves_at_least_two_models_a_side(name: str) -> None:
    """Both the IN and OUT sets need a median and a spread to fit a Gaussian.

    `int(PKEEP * n)` is how many shadows each example is IN, and the remainder
    is how many it is OUT of. One model a side gives a standard deviation of
    zero, which the `+ 1e-30` guard turns into an enormous log-density rather
    than a meaningful score.
    """
    count = shared.shadow_count(get_level(name))
    members = int(shared.PKEEP * count)

    assert members >= 2
    assert count - members >= 2


@pytest.mark.parametrize("name", LEVEL_NAMES)
def test_the_bank_size_reaches_the_spec_and_the_row(name: str) -> None:
    """The count must be recorded, so a reviewer sees which bank produced a row.

    It is part of the shadow bank's `subset_selector`, which keys the checkpoint
    cache, so a smoke bank can never be loaded in place of a full one.
    """
    level = get_level(name).with_defaults(epochs=1)
    spec = shared.shadow_bank_spec(
        level, seed=0, capacity="m1", num_features=4, num_classes=2
    )

    assert str(shared.shadow_count(level)) in spec.subset_selector


def test_full_runs_the_papers_inversion_budget() -> None:
    """`full` is the reproduction; its alpha is the published one."""
    assert data_reconstruction._alpha(_context("full")) == shared.RECONSTRUCTION_ALPHA
    assert shared.RECONSTRUCTION_ALPHA == 3000


@pytest.mark.parametrize("name", ("test", "smoke"))
def test_reduced_levels_invert_for_fewer_steps(name: str) -> None:
    """Each inversion step is a forward and backward pass through the target."""
    assert data_reconstruction._alpha(_context(name)) < shared.RECONSTRUCTION_ALPHA


@pytest.mark.parametrize("name", LEVEL_NAMES)
def test_every_inversion_budget_runs_the_loop_at_least_once(name: str) -> None:
    """A budget of zero would skip the optimization the sub-attack exists to run."""
    assert data_reconstruction._alpha(_context(name)) >= 1


def test_full_runs_the_papers_evasion_chain() -> None:
    """`full` is the reproduction; its PGD chain is the published one."""
    assert shared.evasion_iterations_for(get_level("full")) == shared.EVASION_ITERATIONS
    assert shared.EVASION_ITERATIONS == 40


@pytest.mark.parametrize("name", ("test", "smoke"))
def test_reduced_levels_run_a_shorter_evasion_chain(name: str) -> None:
    """PGD runs per batch over the whole test split, so the count multiplies.

    This one was previously read straight off the constant, so no level reduced
    it at all and even the CPU tier took all 40 steps.
    """
    assert shared.evasion_iterations_for(get_level(name)) < shared.EVASION_ITERATIONS


@pytest.mark.parametrize("name", LEVEL_NAMES)
def test_every_evasion_chain_takes_at_least_one_step(name: str) -> None:
    """A zero-step chain would return the clean input and measure nothing."""
    assert shared.evasion_iterations_for(get_level(name)) >= 1


def test_each_level_gets_its_own_shadow_bank_checkpoint() -> None:
    """Two levels' banks differ in size, so they must not share a cache key."""
    keys = {
        shared.shadow_bank_spec(
            get_level(name).with_defaults(epochs=1),
            seed=0,
            capacity="m1",
            num_features=4,
            num_classes=2,
        ).key()
        for name in LEVEL_NAMES
    }

    assert len(keys) == len(LEVEL_NAMES)
