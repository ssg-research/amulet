"""Uniform entry point for E2, Adversarial Training x Model Ownership.

Registry `e2_advtr_modext`. For each requested (dataset, seed, epsilon) the sweep
trains a clean baseline (once per dataset/seed, reused across budgets),
adversarially trains a defended model at the budget, distils a surrogate from the
defended model, and records both models' clean and robust accuracies and the
surrogate's fidelity:

    python artifact/experiments/e2_advtr_modext/run.py --level test
    python artifact/experiments/e2_advtr_modext/run.py --level full --datasets census,lfw
    python artifact/experiments/e2_advtr_modext/run.py --level full --seeds 0-4 --epsilons 0.01,0.1

`run(...)` is the same path under a callable name, used by the level sweepers and
the tiny end-to-end test. `--level test` substitutes tiny synthetic tabular data
for every dataset, so the fast tier needs no download. The clean and defended
models are separate checkpoints, so every "defended" measurement is the defended
model's, not the clean one's.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse

import torch

from common import run_context, training
from common.cli import parse_seeds
from common.config import LEVEL_NAMES, get_level
from experiments.e2_advtr_modext import train_targets
from experiments.e2_advtr_modext.schemas import CAPACITY, DATASETS, EPSILONS, SCHEMA
from experiments.e2_advtr_modext.train_targets import BATCH_SIZE, PAPER_EPOCHS

EXPERIMENT_ID = "e2_advtr_modext"


def run_cell(
    ctx: run_context.RunContext, dataset: str, epsilon: float, output_dir: Path
) -> list[dict[str, object]]:
    """Run one (dataset, seed, epsilon) cell and append its result row.

    Args:
        ctx: The run context.
        dataset: The dataset name.
        epsilon: The swept perturbation budget.
        output_dir: Directory the result CSV is written under.

    Returns:
        The single row appended, or an empty list if the cell was already recorded.
    """
    from amulet.unauth_model_ownership.metrics import evaluate_extraction
    from common.io import append_row, row_exists

    output = output_dir / f"{EXPERIMENT_ID}.csv"
    key = {
        "exp_id": ctx.seed,
        "dataset": dataset,
        "capacity": CAPACITY,
        "epsilon": epsilon,
    }
    if row_exists(output, SCHEMA, key):
        return []

    started = time.perf_counter()
    bundle, data = train_targets.build_models(ctx, dataset, epsilon)
    batch_size = run_context.batch_for(ctx.level, BATCH_SIZE)
    applied_epsilon = run_context.epsilon_for(ctx.level, epsilon)
    iterations = run_context.pgd_iterations_for(ctx.level)
    test_loader = training.loader_for(data.test_set, batch_size)

    from amulet.utils import get_accuracy

    # Every "defended" number is measured on the defended model, the target
    # accuracy on the clean baseline; they are distinct checkpoints.
    target_test_acc = get_accuracy(bundle.clean, test_loader, ctx.device)
    defended_test_acc = get_accuracy(bundle.defended, test_loader, ctx.device)
    target_robust_acc = training.robust_accuracy(
        bundle.clean, test_loader, ctx.device, batch_size, applied_epsilon, iterations
    )
    defended_robust_acc = training.robust_accuracy(
        bundle.defended,
        test_loader,
        ctx.device,
        batch_size,
        applied_epsilon,
        iterations,
    )
    scores = evaluate_extraction(
        bundle.defended, bundle.stolen, test_loader, ctx.device
    )

    row: dict[str, object] = {
        "exp_id": ctx.seed,
        "dataset": dataset,
        "arch": bundle.defended_spec.arch,
        "capacity": CAPACITY,
        "training_size": ctx.level.train_fraction,
        "epochs": bundle.defended_spec.epochs,
        "batch_size": batch_size,
        "adv_train_fraction": train_targets.ADVERSARY_FRACTION,
        "epsilon": epsilon,
        "step_size": training.step_size_for(applied_epsilon),
        "iterations": iterations,
        "target_test_acc": target_test_acc,
        "defended_test_acc": defended_test_acc,
        "target_robust_acc": target_robust_acc,
        "defended_robust_acc": defended_robust_acc,
        "stolen_test_acc": scores["stolen_accuracy"],
        "fidelity": scores["fidelity"],
        "correct_fidelity": scores["correct_fidelity"],
        "runtime_sec": round(time.perf_counter() - started, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _ = append_row(output, SCHEMA, row)
    return [row]


def run(
    level: str = "full",
    seeds: tuple[int, ...] | None = None,
    datasets: tuple[str, ...] = DATASETS,
    epsilons: tuple[float, ...] = EPSILONS,
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    device: str | None = None,
) -> list[dict[str, object]]:
    """Run E2 at one verification level and return the rows it appended.

    Args:
        level: One of `common.config.LEVEL_NAMES`.
        seeds: Seeds to sweep. None keeps the level's own seeds.
        datasets: Datasets to sweep, a subset of `schemas.DATASETS`.
        epsilons: Budgets to sweep, a subset of `schemas.EPSILONS`.
        output_dir: Directory the result CSV goes in. None keeps the per-level
            default from `default_output_dir`: each level's own `runs/<level>/`
            subtree, so a cheap run never overwrites a `full` run's numbers.
        cache_dir: Checkpoint cache directory. None keeps the per-level default.
        device: Torch device. None picks CUDA when available, else CPU.

    Returns:
        Every row appended by this call. Cells already recorded are skipped.
    """
    config = get_level(level).with_defaults(epochs=PAPER_EPOCHS)
    if seeds is not None:
        config = config.override(seeds=tuple(seeds))

    if config.tiny_model:
        # Tiny CPU tensors spend more time in thread dispatch than arithmetic;
        # one thread is far faster here and does not affect real GPU levels.
        torch.set_num_threads(1)

    resolved_device = (
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    directory = (
        output_dir
        if output_dir is not None
        else run_context.default_output_dir(config, EXPERIMENT_ID)
    )
    directory.mkdir(parents=True, exist_ok=True)
    resolved_cache = (
        cache_dir if cache_dir is not None else run_context.default_cache_dir(config)
    )

    from common import progress

    sweep = [
        (seed, dataset, epsilon)
        for seed in config.seeds
        for dataset in datasets
        for epsilon in epsilons
    ]

    rows: list[dict[str, object]] = []
    ctx: run_context.RunContext | None = None
    current_seed: int | None = None
    for seed, dataset, epsilon in progress.cells(sweep, EXPERIMENT_ID):
        if seed != current_seed:
            training.seed_everything(seed)
            ctx = run_context.RunContext(
                level=config,
                seed=seed,
                device=resolved_device,
                cache_dir=resolved_cache,
            )
            current_seed = seed
        assert ctx is not None
        progress.log(f"[{EXPERIMENT_ID}] {dataset} eps={epsilon:g} seed={seed}")
        rows.extend(run_cell(ctx, dataset, epsilon, directory))
    return rows


def _parse_datasets(text: str) -> tuple[str, ...]:
    """Parse a comma-separated dataset subset, validated against `DATASETS`."""
    if text.strip() == "all":
        return DATASETS
    requested = [piece.strip() for piece in text.split(",") if piece.strip()]
    unknown = [name for name in requested if name not in DATASETS]
    if unknown:
        raise ValueError(
            f"Unknown dataset(s): {', '.join(unknown)}. Choose from: {', '.join(DATASETS)}."
        )
    return tuple(name for name in DATASETS if name in set(requested))


def _parse_epsilons(text: str) -> tuple[float, ...]:
    """Parse a comma-separated budget subset, validated against `EPSILONS`."""
    if text.strip() == "all":
        return EPSILONS
    requested = [float(piece.strip()) for piece in text.split(",") if piece.strip()]
    unknown = [value for value in requested if value not in EPSILONS]
    if unknown:
        known = ", ".join(f"{value:g}" for value in EPSILONS)
        raise ValueError(
            f"Unknown epsilon(s): {', '.join(f'{v:g}' for v in unknown)}. Choose from: {known}."
        )
    return tuple(value for value in EPSILONS if value in set(requested))


def main(argv: list[str] | None = None) -> None:
    """Run E2 from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=str, default="full", choices=LEVEL_NAMES)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Seeds to sweep, e.g. `0` or `0-4`. Default: the level's own seeds.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help=f"Comma-separated subset of: {', '.join(DATASETS)}. Default: all.",
    )
    parser.add_argument(
        "--epsilons",
        type=str,
        default="all",
        help=f"Comma-separated subset of: {', '.join(f'{v:g}' for v in EPSILONS)}. Default: all.",
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
        datasets=_parse_datasets(args.datasets),
        epsilons=_parse_epsilons(args.epsilons),
        device=args.device,
    )


if __name__ == "__main__":
    main()
