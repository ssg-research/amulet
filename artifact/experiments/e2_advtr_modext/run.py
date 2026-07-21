"""Uniform entry point for E2, Adversarial Training x Model Ownership.

Registry `e2_advtr_modext`. For each requested (dataset, seed, epsilon) the
sweep trains a clean baseline $\\modelstd$ (once per dataset/seed, reused across
budgets), adversarially trains a defended model $\\modeldef$ at the budget,
distils a stolen surrogate $\\modelstol$ from $\\modeldef$, and records both
models' clean and robust accuracies and the surrogate's fidelity:

    python artifact/experiments/e2_advtr_modext/run.py --level test
    python artifact/experiments/e2_advtr_modext/run.py --level full --datasets census,lfw
    python artifact/experiments/e2_advtr_modext/run.py --level full --seeds 0-4 --epsilons 0.01,0.1

`run(...)` is the same path under a callable name, used by the level sweepers and
the tiny end-to-end test. `--level test` substitutes tiny synthetic tabular data
for every dataset, so the fast tier needs no download.

The old `advtr_modelext.py:189` overwrote the adversarially-trained model with
the plain target before evaluation. We do not: $\\modelstd$ and $\\modeldef$ are
separate checkpoints, and every "defended" measurement here is $\\modeldef$'s.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from common.config import LEVEL_NAMES, get_level
from experiments import advtr_common as advtr
from experiments.e1_attack_baselines.run import parse_seeds
from experiments.e2_advtr_modext.schemas import CAPACITY, DATASETS, EPSILONS, SCHEMA

if TYPE_CHECKING:
    from amulet.datasets import AmuletDataset
    from common.models import ModelSpec

EXPERIMENT_ID = "e2_advtr_modext"

# The paper trains the E2 victims for 100 epochs (`advtr_modelext.py` default);
# `full` defers to this via `with_defaults`.
PAPER_EPOCHS = 100

# Batch size from the old `advtr_modelext.py` default.
BATCH_SIZE = 256


@dataclass(frozen=True)
class ModelBundle:
    """The three models one E2 cell trains, plus the specs that keyed them.

    Exposed as a seam so a test can confirm the defended model is genuinely the
    adversarially-trained one, distinct from the clean target (plan S5).

    Attributes:
        clean: The clean baseline $\\modelstd$.
        defended: The adversarially-trained $\\modeldef$.
        stolen: The surrogate $\\modelstol$ distilled from $\\modeldef$.
        clean_spec: The spec that keyed `clean`.
        defended_spec: The spec that keyed `defended`.
    """

    clean: nn.Module
    defended: nn.Module
    stolen: nn.Module
    clean_spec: ModelSpec
    defended_spec: ModelSpec


def build_models(
    ctx: advtr.RunContext, dataset: str, epsilon: float, capacity: str = CAPACITY
) -> tuple[ModelBundle, AmuletDataset]:
    """Train (or load) the clean, defended and stolen models for one cell.

    Args:
        ctx: The run context (level, seed, device, cache directory).
        dataset: The dataset name.
        epsilon: The perturbation budget (the swept value, which keys the row).
        capacity: The capacity tier.

    Returns:
        The three models with their specs, and the loaded dataset.
    """
    from amulet.unauth_model_ownership.attacks import ModelExtraction

    data = ctx.data(dataset)
    num_features, num_classes = data.num_features, data.num_classes
    batch_size = advtr.batch_for(ctx.level, BATCH_SIZE)
    applied_epsilon = advtr.epsilon_for(ctx.level, epsilon)
    iterations = advtr.pgd_iterations_for(ctx.level)

    target_set, adversary_set = advtr.dataset_adversary_split(data.train_set, ctx.seed)
    target_loader = advtr.loader_for(target_set, batch_size)
    adversary_loader = advtr.loader_for(adversary_set, batch_size)

    clean_spec = advtr.clean_target_spec(
        ctx.level,
        dataset,
        ctx.seed,
        capacity,
        num_features,
        num_classes,
        batch_size,
        advtr.DATASET_SPLIT_TARGET,
    )
    defended_spec = advtr.defended_target_spec(
        ctx.level,
        dataset,
        ctx.seed,
        capacity,
        num_features,
        num_classes,
        batch_size,
        epsilon,
        advtr.DATASET_SPLIT_TARGET,
    )
    stolen_spec = advtr.stolen_model_spec(
        ctx.level,
        dataset,
        ctx.seed,
        capacity,
        num_features,
        num_classes,
        batch_size,
        epsilon,
        advtr.DATASET_SPLIT_ADVERSARY,
    )

    clean = ctx.get_or_train(
        clean_spec,
        num_features,
        num_classes,
        lambda model: advtr.train_clean(
            model, target_loader, ctx.device, clean_spec.epochs
        ),
    )
    defended = ctx.get_or_train(
        defended_spec,
        num_features,
        num_classes,
        lambda model: advtr.adversarially_train(
            model,
            target_loader,
            ctx.device,
            defended_spec.epochs,
            applied_epsilon,
            iterations,
        ),
    )

    def distil(model: nn.Module) -> nn.Module:
        # The surrogate is distilled from the DEFENDED model, never the clean
        # target: the adversary steals the robust model (plan S5, confirmed
        # against the committed CSVs where stolen accuracy tracks $\\modeldef$).
        extraction = ModelExtraction(
            defended,
            model,
            torch.optim.Adam(model.parameters(), lr=1e-3),
            adversary_loader,
            ctx.device,
            stolen_spec.epochs,
            loss_type="mse",
        )
        return extraction.attack()

    stolen = ctx.get_or_train(stolen_spec, num_features, num_classes, distil)

    return (
        ModelBundle(clean, defended, stolen, clean_spec, defended_spec),
        data,
    )


def run_cell(
    ctx: advtr.RunContext, dataset: str, epsilon: float, output_dir: Path
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

    bundle, data = build_models(ctx, dataset, epsilon)
    batch_size = advtr.batch_for(ctx.level, BATCH_SIZE)
    applied_epsilon = advtr.epsilon_for(ctx.level, epsilon)
    iterations = advtr.pgd_iterations_for(ctx.level)
    test_loader = advtr.loader_for(data.test_set, batch_size)

    from amulet.utils import get_accuracy

    # Every "defended" number is measured on $\\modeldef$; the clean baseline
    # test accuracy is $\\modelstd$'s. Were these the same object (the old bug),
    # the two test accuracies would be identical by construction.
    target_test_acc = get_accuracy(bundle.clean, test_loader, ctx.device)
    defended_test_acc = get_accuracy(bundle.defended, test_loader, ctx.device)
    target_robust_acc = advtr.robust_accuracy(
        bundle.clean, test_loader, ctx.device, batch_size, applied_epsilon, iterations
    )
    defended_robust_acc = advtr.robust_accuracy(
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
        "adv_train_fraction": advtr.ADVERSARY_FRACTION,
        "epsilon": epsilon,
        "step_size": advtr.step_size_for(applied_epsilon),
        "iterations": iterations,
        "target_test_acc": target_test_acc,
        "defended_test_acc": defended_test_acc,
        "target_robust_acc": target_robust_acc,
        "defended_robust_acc": defended_robust_acc,
        "stolen_test_acc": scores["stolen_accuracy"],
        "fidelity": scores["fidelity"],
        "correct_fidelity": scores["correct_fidelity"],
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
            default (committed results for `full`, a level subdir for `smoke`, a
            temporary directory for `test`).
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
        else advtr.default_output_dir(config, EXPERIMENT_ID)
    )
    directory.mkdir(parents=True, exist_ok=True)
    resolved_cache = (
        cache_dir if cache_dir is not None else advtr.default_cache_dir(config)
    )

    rows: list[dict[str, object]] = []
    for seed in config.seeds:
        advtr.seed_everything(seed)
        ctx = advtr.RunContext(
            level=config, seed=seed, device=resolved_device, cache_dir=resolved_cache
        )
        for dataset in datasets:
            for epsilon in epsilons:
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
