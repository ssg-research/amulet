"""Uniform entry point for E3, Adversarial Training x Attribute Inference.

Registry `e3_advtr_attrinf`. For each dataset the sweep trains a clean baseline
$\\modelstd$ once, runs attribute inference against it for the baseline row, then
for each budget adversarially trains a $\\modeldef$, records both models' robust
accuracies, and runs attribute inference against $\\modeldef$:

    python artifact/experiments/e3_advtr_attrinf/run.py --level test
    python artifact/experiments/e3_advtr_attrinf/run.py --level full --datasets census
    python artifact/experiments/e3_advtr_attrinf/run.py --level full --seeds 0-4 --epsilons 0.01,0.1

`run(...)` is the same path under a callable name. `--level test` substitutes
tiny synthetic tabular data with two sensitive columns for every dataset.

Both sensitive attributes are inferred together (census: race, sex; lfw: race,
gender). The old `advtr_attrinf.py` ran inference against the plain target with
adversarial training defaulted off; here the epsilon rows run it against the
adversarially-trained $\\modeldef$, and $\\modelstd$/$\\modeldef$ are distinct
checkpoints (plan S5).
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
from experiments.e3_advtr_attrinf.schemas import (
    BASELINE_EPSILON,
    CAPACITY,
    DATASETS,
    EPSILONS,
    SCHEMA,
)

if TYPE_CHECKING:
    from amulet.datasets import AmuletDataset
    from common.models import ModelSpec
    from experiments.advtr_common import AdversarySplit

EXPERIMENT_ID = "e3_advtr_attrinf"

# Target training budget for the E3 datasets; `full` defers to this. The old
# `advtr_attrinf.py` default (200) was tuned for a CelebA ResNet, not the tabular
# census/lfw targets here, so we use the 100 the paper states for its main runs.
PAPER_EPOCHS = 100

# Batch size from the old `advtr_attrinf.py` default.
BATCH_SIZE = 128

# The two sensitive attributes the attack infers, in the order the reference
# table's columns appear. Attribute index 0 is race for both datasets; index 1
# is census's sex and lfw's gender, which the table labels Sex.
_RACE_INDEX = 0
_SEX_INDEX = 1


@dataclass(frozen=True)
class AttributeScores:
    """Attribute-inference results for one model, both attributes at once."""

    acc_race: float
    auc_race: float
    acc_sex: float
    auc_sex: float


def clean_target(
    ctx: advtr.RunContext, dataset: str, capacity: str = CAPACITY
) -> tuple[nn.Module, AdversarySplit, AmuletDataset, ModelSpec]:
    """Train (or load) the clean baseline $\\modelstd$ and the adversary split.

    Args:
        ctx: The run context.
        dataset: The dataset name (must carry NumPy views and sensitive attributes).
        capacity: The capacity tier.

    Returns:
        The clean model, the numpy adversary split, the dataset, and the spec
        that keyed the model.
    """
    data = ctx.data(dataset)
    split = advtr.adversary_split(data, ctx.seed)
    batch_size = advtr.batch_for(ctx.level, BATCH_SIZE)
    spec = advtr.clean_target_spec(
        ctx.level,
        dataset,
        ctx.seed,
        capacity,
        data.num_features,
        data.num_classes,
        batch_size,
        advtr.ARRAY_SPLIT_TARGET,
    )
    loader = advtr.loader_for(split.target_set, batch_size)
    model = ctx.get_or_train(
        spec,
        data.num_features,
        data.num_classes,
        lambda m: advtr.train_clean(m, loader, ctx.device, spec.epochs),
    )
    return model, split, data, spec


def defended_target(
    ctx: advtr.RunContext,
    dataset: str,
    epsilon: float,
    split: AdversarySplit,
    data: AmuletDataset,
    capacity: str = CAPACITY,
) -> tuple[nn.Module, ModelSpec]:
    """Train (or load) the adversarially-trained $\\modeldef$ at one budget."""
    batch_size = advtr.batch_for(ctx.level, BATCH_SIZE)
    applied_epsilon = advtr.epsilon_for(ctx.level, epsilon)
    iterations = advtr.pgd_iterations_for(ctx.level)
    spec = advtr.defended_target_spec(
        ctx.level,
        dataset,
        ctx.seed,
        capacity,
        data.num_features,
        data.num_classes,
        batch_size,
        epsilon,
        advtr.ARRAY_SPLIT_TARGET,
    )
    loader = advtr.loader_for(split.target_set, batch_size)
    model = ctx.get_or_train(
        spec,
        data.num_features,
        data.num_classes,
        lambda m: advtr.adversarially_train(
            m, loader, ctx.device, spec.epochs, applied_epsilon, iterations
        ),
    )
    return model, spec


def infer_attributes(
    model: nn.Module,
    split: AdversarySplit,
    data: AmuletDataset,
    batch_size: int,
    device: str,
) -> AttributeScores:
    """Run attribute inference against one model, scoring both attributes.

    Args:
        model: The target to attack.
        split: The adversary's half (its features and sensitive labels).
        data: The dataset, for the test features and sensitive labels.
        batch_size: Batch size for querying the target.
        device: Device to run on.

    Returns:
        Balanced accuracy (as a percentage) and AUC for race and for sex.

    Raises:
        ValueError: If the dataset carries no test features or attributes.
    """
    from amulet.attribute_inference.attacks import DudduCIKM2022
    from amulet.attribute_inference.metrics import evaluate_attribute_inference

    if data.x_test is None or data.z_test is None:
        raise ValueError("Attribute inference needs the test features and attributes.")

    attack = DudduCIKM2022(
        model, split.adversary_x, data.x_test, split.adversary_z, batch_size, device
    )
    metrics = evaluate_attribute_inference(data.z_test, attack.attack())
    return AttributeScores(
        acc_race=metrics[_RACE_INDEX]["attack_accuracy"] * 100,
        auc_race=metrics[_RACE_INDEX]["auc_score"],
        acc_sex=metrics[_SEX_INDEX]["attack_accuracy"] * 100,
        auc_sex=metrics[_SEX_INDEX]["auc_score"],
    )


def _leading(
    ctx: advtr.RunContext,
    dataset: str,
    spec: ModelSpec,
    epsilon: float,
    applied_epsilon: float,
    data: AmuletDataset,
) -> dict[str, object]:
    """Fill the columns identifying which cell and model a row measured."""
    sensitive = data.sensitive_columns or ["", ""]
    return {
        "exp_id": ctx.seed,
        "dataset": dataset,
        "arch": spec.arch,
        "capacity": CAPACITY,
        "training_size": ctx.level.train_fraction,
        "epochs": spec.epochs,
        "batch_size": advtr.batch_for(ctx.level, BATCH_SIZE),
        "adv_train_fraction": advtr.ADVERSARY_FRACTION,
        "epsilon": epsilon,
        "step_size": advtr.step_size_for(applied_epsilon),
        "iterations": advtr.pgd_iterations_for(ctx.level),
        "sensitive_attr_1": sensitive[0],
        "sensitive_attr_2": sensitive[1],
    }


def run_dataset(
    ctx: advtr.RunContext,
    dataset: str,
    epsilons: tuple[float, ...],
    output_dir: Path,
) -> list[dict[str, object]]:
    """Run one dataset's baseline row and every budget row, appending each.

    The clean $\\modelstd$ is trained once and reused for the baseline attribute
    inference and for every budget's undefended robust accuracy.

    Args:
        ctx: The run context.
        dataset: The dataset name.
        epsilons: The budgets to sweep.
        output_dir: Directory the result CSV is written under.

    Returns:
        Every row appended for this dataset.
    """
    from amulet.utils import get_accuracy
    from common.io import append_row, row_exists

    output = output_dir / f"{EXPERIMENT_ID}.csv"
    batch_size = advtr.batch_for(ctx.level, BATCH_SIZE)
    rows: list[dict[str, object]] = []

    needed = [BASELINE_EPSILON, *epsilons]
    if all(
        row_exists(
            output,
            SCHEMA,
            {
                "exp_id": ctx.seed,
                "dataset": dataset,
                "capacity": CAPACITY,
                "epsilon": eps,
            },
        )
        for eps in needed
    ):
        return rows

    clean, split, data, clean_spec = clean_target(ctx, dataset)
    test_loader = advtr.loader_for(data.test_set, batch_size)

    baseline_key = {
        "exp_id": ctx.seed,
        "dataset": dataset,
        "capacity": CAPACITY,
        "epsilon": BASELINE_EPSILON,
    }
    if not row_exists(output, SCHEMA, baseline_key):
        clean_scores = infer_attributes(clean, split, data, batch_size, ctx.device)
        row: dict[str, object] = {
            **_leading(
                ctx, dataset, clean_spec, BASELINE_EPSILON, BASELINE_EPSILON, data
            ),
            "model_role": "baseline",
            "test_acc": get_accuracy(clean, test_loader, ctx.device),
            # No defended model at baseline, so the robust columns are blank.
            "target_robust_acc": "",
            "defended_robust_acc": "",
            "acc_att_race": clean_scores.acc_race,
            "auc_race": clean_scores.auc_race,
            "acc_att_sex": clean_scores.acc_sex,
            "auc_sex": clean_scores.auc_sex,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _ = append_row(output, SCHEMA, row)
        rows.append(row)

    for epsilon in epsilons:
        key = {
            "exp_id": ctx.seed,
            "dataset": dataset,
            "capacity": CAPACITY,
            "epsilon": epsilon,
        }
        if row_exists(output, SCHEMA, key):
            continue
        applied_epsilon = advtr.epsilon_for(ctx.level, epsilon)
        iterations = advtr.pgd_iterations_for(ctx.level)
        defended, defended_spec = defended_target(ctx, dataset, epsilon, split, data)
        defended_scores = infer_attributes(
            defended, split, data, batch_size, ctx.device
        )
        row = {
            **_leading(ctx, dataset, defended_spec, epsilon, applied_epsilon, data),
            "model_role": "defended",
            "test_acc": get_accuracy(defended, test_loader, ctx.device),
            "target_robust_acc": advtr.robust_accuracy(
                clean, test_loader, ctx.device, batch_size, applied_epsilon, iterations
            ),
            "defended_robust_acc": advtr.robust_accuracy(
                defended,
                test_loader,
                ctx.device,
                batch_size,
                applied_epsilon,
                iterations,
            ),
            "acc_att_race": defended_scores.acc_race,
            "auc_race": defended_scores.auc_race,
            "acc_att_sex": defended_scores.acc_sex,
            "auc_sex": defended_scores.auc_sex,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _ = append_row(output, SCHEMA, row)
        rows.append(row)

    return rows


def run(
    level: str = "full",
    seeds: tuple[int, ...] | None = None,
    datasets: tuple[str, ...] = DATASETS,
    epsilons: tuple[float, ...] = EPSILONS,
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    device: str | None = None,
) -> list[dict[str, object]]:
    """Run E3 at one verification level and return the rows it appended.

    Args:
        level: One of `common.config.LEVEL_NAMES`.
        seeds: Seeds to sweep. None keeps the level's own seeds.
        datasets: Datasets to sweep, a subset of `schemas.DATASETS`.
        epsilons: Budgets to sweep, a subset of `schemas.EPSILONS`.
        output_dir: Directory the result CSV goes in. None keeps the per-level default.
        cache_dir: Checkpoint cache directory. None keeps the per-level default.
        device: Torch device. None picks CUDA when available, else CPU.

    Returns:
        Every row appended by this call. Cells already recorded are skipped.
    """
    config = get_level(level).with_defaults(epochs=PAPER_EPOCHS)
    if seeds is not None:
        config = config.override(seeds=tuple(seeds))

    if config.tiny_model:
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
            rows.extend(run_dataset(ctx, dataset, epsilons, directory))
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
    """Run E3 from the command line."""
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
