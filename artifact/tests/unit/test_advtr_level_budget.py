"""E2/E3's level budget: only `full` runs the paper's PGD chain.

PGD's iteration count is the innermost loop in the artifact. Adversarial
training pays it on every batch of every epoch, and the evasion measurement pays
it again across the test split, so it multiplies with the epochs, the training
fraction and the test fraction a reduced level has already cut. Every step is
the same forward, backward and clip: the count tightens the perturbation without
executing a line a shorter chain would miss.

`smoke` therefore runs PGD-7, the standard configuration from the adversarial
training literature rather than an arbitrary shrink, and `full` keeps the
paper's 40.

No data, no model, no GPU: a function of the level preset alone.
"""

from __future__ import annotations

import pytest

from common.config import LEVEL_NAMES, get_level
from experiments import shared_targets as targets


def test_full_runs_the_papers_pgd_chain() -> None:
    """`full` is the reproduction; its chain is the published one."""
    assert targets.pgd_iterations_for(get_level("full")) == targets.PGD_ITERATIONS
    assert targets.PGD_ITERATIONS == 40


@pytest.mark.parametrize("name", ("test", "smoke"))
def test_reduced_levels_run_a_shorter_pgd_chain(name: str) -> None:
    """The innermost loop must shrink with everything else the level cut."""
    assert targets.pgd_iterations_for(get_level(name)) < targets.PGD_ITERATIONS


@pytest.mark.parametrize("name", LEVEL_NAMES)
def test_every_pgd_chain_takes_at_least_one_step(name: str) -> None:
    """A zero-step chain would leave the input clean and measure nothing."""
    assert targets.pgd_iterations_for(get_level(name)) >= 1


def test_both_pgd_users_agree_on_a_reduced_chain() -> None:
    """E1's evasion and E2/E3's adversarial training both run PGD.

    They are configured in separate modules, so nothing but this test stops one
    from being reduced while the other quietly keeps the paper's chain.
    """
    from experiments.e1_attack_baselines import shared

    level = get_level("smoke")

    assert targets.pgd_iterations_for(level) == shared.evasion_iterations_for(level)
