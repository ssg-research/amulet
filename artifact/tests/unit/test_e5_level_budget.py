"""E5's level budget: `smoke` runs one cell, `full` runs the paper's grid.

E5 is the only experiment whose sweep buys no path coverage. Its five poison
rates (and DP's two privacy budgets) drive the identical code over the identical
dataset and the identical model; the rate is a scalar, not a branch. Each extra
cell costs a LoRA fine-tune of a 3B-parameter Llama, so a `smoke` level that
kept the full grid would take hours and dwarf E1-E4, which together finish in
minutes. E1-E4 keep their sweeps because those span datasets and architectures
and so do cover distinct paths.

These tests pin that decision at both ends: `smoke` collapses to a single cell,
`full` is untouched, and the value `smoke` keeps is one the paper actually ran
rather than an invented one. No data, no model, no GPU: `apply_level` mutates an
argparse namespace and that is all this inspects.
"""

from __future__ import annotations

import pytest

from common.config import get_level
from experiments.e5_textbadnets import dp, onion


def _levelled(module, level_name: str):
    """Parse the module's defaults, then apply one level to them."""
    args = module.parse_args([])
    config = get_level(level_name).with_defaults(epochs=module.PAPER_EPOCHS)
    module.apply_level(args, config, seed=0)
    return args


def _rates(args) -> list[float]:
    return [float(piece) for piece in str(args.poisoned_portions).split(",")]


@pytest.mark.parametrize("module", (onion, dp), ids=("onion", "dp"))
def test_smoke_runs_a_single_poison_rate(module) -> None:
    """One rate at smoke: each extra rate is two more Llama fine-tunes."""
    assert len(_rates(_levelled(module, "smoke"))) == 1


def test_smoke_runs_a_single_privacy_budget() -> None:
    """DP's epsilons multiply the grid on top of the poison rates."""
    budgets = str(_levelled(dp, "smoke").target_epsilons).split(",")

    assert len(budgets) == 1


@pytest.mark.parametrize("module", (onion, dp), ids=("onion", "dp"))
def test_the_smoke_rate_is_one_the_paper_ran(module) -> None:
    """The kept cell must come from the paper's grid, not be invented for smoke."""
    paper_rates = _rates(_levelled(module, "full"))

    assert _rates(_levelled(module, "smoke"))[0] in paper_rates


def test_the_smoke_rate_poisons_at_least_one_sentence() -> None:
    """A rate that rounds to zero poisoned sentences would measure nothing.

    Smoke reads a fraction of SST-2, so the paper's smallest rate (0.0001) puts
    `round(0.0001 * 6735) = 1` sentence in a tenth of the corpus and less in
    anything smaller. Whichever rate smoke keeps has to survive that arithmetic.
    """
    args = _levelled(onion, "smoke")

    assert round(_rates(args)[0] * args.max_train_samples) >= 1


@pytest.mark.parametrize("module", (onion, dp), ids=("onion", "dp"))
def test_full_keeps_the_whole_paper_grid(module) -> None:
    """`full` is the reproduction and must not inherit smoke's single cell."""
    args = _levelled(module, "full")

    assert len(_rates(args)) > 1
    # -1 is the runners' sentinel for "read the whole split".
    assert args.max_train_samples == -1
    assert args.max_test_samples == -1


def test_full_keeps_both_privacy_budgets() -> None:
    assert len(str(_levelled(dp, "full").target_epsilons).split(",")) == 2


@pytest.mark.parametrize("module", (onion, dp), ids=("onion", "dp"))
def test_smoke_caps_data_at_a_small_absolute_slice(module) -> None:
    """Smoke reads a few hundred records, not the level's 10% of a 67k corpus.

    Ten percent of SST-2's train split is ~6735 sentences, ~420 fine-tune steps
    on the real LM per condition; the cap keeps smoke to a slice that exercises
    every path in minutes. Both splits shrink (trimming only the training one was
    the `load_data` bug this session fixed), and `full` reads the whole corpus.
    """
    from experiments.e5_textbadnets.llm_backdoor_common import (
        SMOKE_MAX_TEST_SAMPLES,
        SMOKE_MAX_TRAIN_SAMPLES,
    )

    smoke = _levelled(module, "smoke")

    assert smoke.max_train_samples == SMOKE_MAX_TRAIN_SAMPLES
    assert smoke.max_test_samples == SMOKE_MAX_TEST_SAMPLES
    # A slice, not a fraction: an order of magnitude below the level's old 10%.
    assert smoke.max_train_samples < 0.1 * module.SST2_TRAIN_SIZE
    # Still enough that the poison rate inserts at least one trigger.
    assert round(0.05 * smoke.max_train_samples) >= 1


@pytest.mark.parametrize("module", (onion, dp), ids=("onion", "dp"))
def test_smoke_swaps_in_the_smaller_real_model(module) -> None:
    """Smoke fine-tunes the 1.1B real Llama, not the 3B paper target.

    The dominant E5 cost is the model, not the cell count, so smoke drops to a
    smaller pretrained causal LM while `full` keeps the paper's 3B target.
    """
    from experiments.e5_textbadnets.llm_backdoor_common import SMOKE_MODEL_NAME

    smoke = _levelled(module, "smoke")
    full = _levelled(module, "full")

    assert smoke.model_name == SMOKE_MODEL_NAME
    assert full.model_name == "meta-llama/Llama-3.2-3B"
    assert smoke.model_name != full.model_name


def test_onion_smoke_reference_lm_matches_its_target() -> None:
    """ONION scores perplexity with the target's own base LM, so when smoke swaps
    the target it must swap the reference too; `reference_model` otherwise keeps
    the paper default it was given in `parse_args`, before `apply_level` ran."""
    smoke = _levelled(onion, "smoke")

    assert smoke.reference_model == smoke.model_name
