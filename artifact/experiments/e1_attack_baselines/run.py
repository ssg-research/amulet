"""Uniform entry point for E1, the attack baselines (registry `e1_attack_baselines`).

Sweeps the requested attacks across the requested VGG capacities on CelebA, one
seed at a time, through the shared content-addressed model cache:

    python artifact/experiments/e1_attack_baselines/run.py --level test
    python artifact/experiments/e1_attack_baselines/run.py --level full --attacks evasion,poisoning
    python artifact/experiments/e1_attack_baselines/run.py --level full --seeds 0-9 --capacities m1

`run(level, seeds, attacks, capacities, output_dir, cache_dir)` is the same path
under a callable name, used by the level sweepers and the tiny end-to-end test.
This runner exposes only what every experiment exposes, so the sweepers treat all
five alike.

CelebA is a large download `scarab.utils.load_data` handles on first use;
`--level test` never touches it, substituting tiny synthetic tensors so the fast
tier runs anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from typing import TYPE_CHECKING

import torch

from common.cli import parse_seeds
from common.config import LEVEL_NAMES, get_level
from common.training import seed_everything
from experiments.e1_attack_baselines import context
from experiments.e1_attack_baselines.schemas import (
    ATTACKS,
    CAPACITIES,
    SINGLE_COLUMN_ATTACKS,
)

if TYPE_CHECKING:
    from types import ModuleType

EXPERIMENT_ID = "e1_attack_baselines"

# The paper's epoch count, which the `full` level defers to.
PAPER_EPOCHS = context.PAPER_EPOCHS


def _parse_selection(text: str, allowed: tuple[str, ...], noun: str) -> tuple[str, ...]:
    """Parse and validate a comma-separated subset of `allowed`.

    Args:
        text: Comma-separated names, or `"all"`.
        allowed: The full set of valid names, in canonical order.
        noun: What the names are, for the error message.

    Returns:
        The requested names in canonical order, without duplicates.

    Raises:
        ValueError: If any requested name is not in `allowed`.
    """
    if text.strip() == "all":
        return allowed
    requested = [piece.strip() for piece in text.split(",") if piece.strip()]
    unknown = [name for name in requested if name not in allowed]
    if unknown:
        raise ValueError(
            f"Unknown {noun}: {', '.join(unknown)}. Choose from: {', '.join(allowed)}."
        )
    return tuple(name for name in allowed if name in set(requested))


def _capacities_for(attack: str, capacities: tuple[str, ...]) -> tuple[str, ...]:
    """Restrict a single-column attack to the first requested capacity.

    Membership inference attacks an overfit ResNet, not the VGG the columns
    name, so the paper reports it in the VGG11 column alone. Running it for every
    capacity would train several overfit ResNets whose numbers the table has no
    place for.

    Args:
        attack: The sub-attack ID.
        capacities: The capacities requested for the sweep.

    Returns:
        Every requested capacity, or just the first for a single-column attack.
    """
    if attack in SINGLE_COLUMN_ATTACKS:
        return capacities[:1]
    return capacities


def run(
    level: str = "full",
    seeds: tuple[int, ...] | None = None,
    attacks: tuple[str, ...] = ATTACKS,
    capacities: tuple[str, ...] = CAPACITIES,
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    device: str | None = None,
) -> list[dict[str, object]]:
    """Run E1 at one verification level and return the rows it appended.

    Args:
        level: One of `common.config.LEVEL_NAMES`.
        seeds: Seeds to sweep. None keeps the level's own seeds.
        attacks: Sub-attacks to run, a subset of `schemas.ATTACKS`.
        capacities: VGG capacities to sweep, a subset of `schemas.CAPACITIES`.
        output_dir: Directory the result CSVs go in. None keeps the per-level
            default from `default_output_dir`: each level's own `runs/<level>/`
            subtree, so a cheap run never overwrites a `full` run's numbers.
        cache_dir: Checkpoint cache directory. None keeps the per-level default
            from `default_cache_dir`: this level's own `.model_cache/<level>/`
            for `smoke`/`full`, a temporary one for `test`.
        device: Torch device. None picks CUDA when available, else CPU.

    Returns:
        Every row appended by this call, across the requested seeds, attacks and
        capacities. Cells already recorded are skipped, so a resumed sweep
        returns only what it added.
    """
    config = get_level(level).with_defaults(epochs=PAPER_EPOCHS)
    if seeds is not None:
        config = config.override(seeds=tuple(seeds))

    if config.tiny_model:
        # Tiny CPU tensors spend far more time in thread dispatch than in
        # arithmetic; one thread is dramatically faster here (seconds, not
        # minutes) and does not affect the real GPU levels.
        torch.set_num_threads(1)

    resolved_device = (
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    directory = (
        output_dir if output_dir is not None else context.default_output_dir(config)
    )
    directory.mkdir(parents=True, exist_ok=True)
    resolved_cache = (
        cache_dir if cache_dir is not None else context.default_cache_dir(config)
    )

    from common import progress

    modules = _load_modules(attacks)
    sweep = [
        (seed, attack, capacity)
        for seed in config.seeds
        for attack in attacks
        for capacity in _capacities_for(attack, capacities)
    ]

    rows: list[dict[str, object]] = []
    ctx: context.RunContext | None = None
    current_seed: int | None = None
    for seed, attack, capacity in progress.cells(sweep, EXPERIMENT_ID):
        if seed != current_seed:
            seed_everything(seed)
            ctx = context.RunContext(
                level=config,
                seed=seed,
                device=resolved_device,
                cache_dir=resolved_cache,
            )
            current_seed = seed
        assert ctx is not None
        progress.log(f"[{EXPERIMENT_ID}] {attack} {capacity} seed={seed}")
        rows.extend(modules[attack].run_cell(ctx, capacity, directory))
    return rows


def _load_modules(attacks: tuple[str, ...]) -> dict[str, ModuleType]:
    """Import each requested sub-attack module on demand.

    Imported here, not at module scope, so `--help` and the registry lookup do
    not pay for torch and the attack libraries.
    """
    import importlib

    return {
        attack: importlib.import_module(
            f"experiments.e1_attack_baselines.attacks.{attack}"
        )
        for attack in attacks
    }


def main(argv: list[str] | None = None) -> None:
    """Run E1 from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=str, default="full", choices=LEVEL_NAMES)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Seeds to sweep, e.g. `0` or `0-9`. Default: the level's own seeds.",
    )
    parser.add_argument(
        "--attacks",
        type=str,
        default="all",
        help=f"Comma-separated subset of: {', '.join(ATTACKS)}. Default: all.",
    )
    parser.add_argument(
        "--capacities",
        type=str,
        default="all",
        help=f"Comma-separated subset of: {', '.join(CAPACITIES)}. Default: all.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device. Default: cuda when available, else cpu.",
    )
    args = parser.parse_args(argv)
    _ = run(
        level=args.level,
        seeds=None if args.seeds is None else parse_seeds(args.seeds),
        attacks=_parse_selection(args.attacks, ATTACKS, "attack"),
        capacities=_parse_selection(args.capacities, CAPACITIES, "capacity"),
        device=args.device,
    )


if __name__ == "__main__":
    main()
