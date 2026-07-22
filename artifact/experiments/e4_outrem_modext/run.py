"""Uniform entry point for E4, Outlier Removal x Model Ownership.

Registry `e4_outrem_modext`. For each requested (dataset, seed, percent) the
sweep trains a clean baseline $\\modelstd$ (once per dataset/seed, reused across
every removal percentage), applies kNN-Shapley outlier removal at that
percentage and retrains to get a defended model $\\modeldef$, distils a stolen
surrogate $\\modelstol$ from $\\modeldef$, and records the defended model's test
accuracy and the surrogate's accuracy, fidelity and correct fidelity:

    python artifact/experiments/e4_outrem_modext/run.py --level test
    python artifact/experiments/e4_outrem_modext/run.py --level full --datasets census,lfw
    python artifact/experiments/e4_outrem_modext/run.py --level full --seeds 0-4 --percents 10,20

`run(...)` is the same path under a callable name, used by the level sweepers and
the tiny end-to-end test. `--level test` substitutes tiny synthetic tabular data
for every dataset, so the fast tier needs no download.

**Baseline shared with E2.** E4 has no original script; it composes the outlier
removal defense with model extraction, which no old script did together. Its
clean baseline is a clean model-extraction target on the same four datasets as
E2, built from the *identical* spec (`clean_target_spec` with the 50/50
dataset-level split selector, Adam at 1e-3, 100 epochs, batch 256). The
content-addressed cache therefore stores one $\\modelstd$ checkpoint serving both
experiments (plan S6, S13). The removal percentage `0` is that clean baseline: no
outliers are removed, so $\\modeldef$ is $\\modelstd$ and the surrogate is stolen
from the clean model. Every `percent > 0` model encodes the percentage in its
optimizer recipe, so it is a distinct checkpoint that can never be confused with
the baseline or with an E2 defended model (which encodes an epsilon instead).
"""

from __future__ import annotations

import copy
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from amulet.datasets import AmuletDataset
from common.config import LEVEL_NAMES, get_level
from experiments import advtr_common as advtr
from experiments.e1_attack_baselines.run import parse_seeds
from experiments.e4_outrem_modext.schemas import CAPACITY, DATASETS, PERCENTS, SCHEMA

if TYPE_CHECKING:
    from common.models import ModelSpec

EXPERIMENT_ID = "e4_outrem_modext"

# The paper trains the E4 targets for 100 epochs and retrains for 100 more after
# removal (the old `run_model_extraction.py` / `run_outlier_removal.py` default);
# `full` defers to this via `with_defaults`.
PAPER_EPOCHS = 100

# Batch size from the old extraction/outlier-removal scripts' shared default.
BATCH_SIZE = 256

# The `test`-level synthetic stand-in for census/lfw/fmnist/cifar. It is E4's own
# rather than the shared `tiny_tabular_dataset`, because kNN-Shapley outlier
# removal has nothing to remove from the perfectly separable shared stand-in: its
# influence scores come out constant, the score normalisation divides by zero,
# and the "cleaned" set is empty. A fraction of the training labels are therefore
# flipped so genuine mislabelled outliers exist to score and remove, while the
# test set stays clean as the held-out reference the scores are computed against.
TINY_TRAIN_SIZE = 80
TINY_TEST_SIZE = 40
TINY_NUM_FEATURES = 8
TINY_NUM_CLASSES = 2
TINY_OUTLIER_FRACTION = 0.15


def tiny_outrem_dataset(
    seed: int,
    num_features: int = TINY_NUM_FEATURES,
    num_classes: int = TINY_NUM_CLASSES,
) -> AmuletDataset:
    """Build E4's `test`-level stand-in: separable tabular data with outliers.

    Each record sits in its class's own intensity band so the label is learnable,
    but a seeded fraction of the *training* labels are flipped, planting genuine
    mislabelled outliers for kNN-Shapley to find and drop. The test set is left
    clean, since it is the held-out reference the influence scores score against.

    Args:
        seed: Seed for the generator, so two runs build identical data.
        num_features: Number of input features.
        num_classes: Number of label classes.

    Returns:
        A tabular dataset with `train_set`/`test_set` and the `x_*`/`y_*`/`z_*`
        arrays populated and index-aligned. Training labels carry the outliers.
    """
    generator = np.random.default_rng(seed)

    def split(size: int, corrupt: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        labels = np.arange(size) % num_classes
        # Features encode the *true* class; flipping the label afterwards is what
        # makes a record an outlier (its features disagree with its label).
        features = np.clip(
            generator.random((size, num_features), dtype=np.float32) * 0.2
            + 0.7 * labels[:, None].astype(np.float32),
            0.0,
            1.0,
        )
        labels = labels.astype(np.int64)
        if corrupt:
            labels = labels.copy()
            count = max(1, int(TINY_OUTLIER_FRACTION * size))
            outliers = generator.choice(size, size=count, replace=False)
            labels[outliers] = (num_classes - 1) - labels[outliers]
        indices = np.arange(size)
        sensitive = np.stack([(indices // 2) % 2, (indices // 3) % 2], axis=1).astype(
            np.int64
        )
        return features, labels, sensitive

    x_train, y_train, z_train = split(TINY_TRAIN_SIZE, corrupt=True)
    x_test, y_test, z_test = split(TINY_TEST_SIZE, corrupt=False)

    return AmuletDataset(
        train_set=TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        test_set=TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
        num_features=num_features,
        num_classes=num_classes,
        modality="tabular",
        sensitive_columns=["attr_1", "attr_2"],
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        z_train=z_train,
        z_test=z_test,
    )


def outrem_recipe(percent: int) -> str:
    """Return the optimizer-recipe string for an outlier-removed $\\modeldef$.

    The removal percentage is baked into the string, which is the cache's
    contract (plan S6): two defended specs that differ only in the percentage get
    different keys and cannot share a checkpoint, and none can collide with the
    clean baseline (Adam alone) or with an E2 defended model (which encodes an
    epsilon). A `percent == 0` model is not built through this recipe at all; it
    is the shared clean baseline.

    Args:
        percent: The percentage of training outliers removed before retraining.

    Returns:
        A short stable recipe name encoding the removal percentage.
    """
    return f"outrem_knn_shapley_p{percent}_adam_lr1e-3"


def stolen_recipe(percent: int) -> str:
    """Return the recipe string for a surrogate distilled from that $\\modeldef$.

    A distilled model's weights depend on the model it was distilled from, which
    is not a `ModelSpec` field, so the source's removal percentage is carried in
    the recipe. A surrogate stolen from a model at a different removal percentage
    therefore cannot reuse this checkpoint, and neither can an E2 surrogate
    (whose recipe names an epsilon).

    Args:
        percent: The removal percentage of the defended target the surrogate imitates.

    Returns:
        A short stable recipe name encoding the distillation source.
    """
    return f"modext_mse_from_outrem_p{percent}_adam_lr1e-3"


@dataclass(frozen=True)
class ModelBundle:
    """The models one E4 cell trains, plus the specs that keyed them.

    Exposed as a seam so a test can confirm the shared baseline and the distinct
    outlier-removed checkpoints (plan S6, S13).

    Attributes:
        clean: The clean baseline $\\modelstd$, shared with E2.
        defended: The outlier-removed $\\modeldef$; the same object as `clean` at
            `percent == 0`, a distinct retrain otherwise.
        stolen: The surrogate $\\modelstol$ distilled from `defended`.
        clean_spec: The spec that keyed `clean` (identical to E2's clean baseline).
        defended_spec: The spec that keyed `defended`.
        stolen_spec: The spec that keyed `stolen`.
    """

    clean: nn.Module
    defended: nn.Module
    stolen: nn.Module
    clean_spec: ModelSpec
    defended_spec: ModelSpec
    stolen_spec: ModelSpec


def clean_baseline_spec(
    ctx: advtr.RunContext,
    dataset: str,
    num_features: int,
    num_classes: int,
    batch_size: int,
    capacity: str = CAPACITY,
) -> ModelSpec:
    """Describe E4's clean baseline $\\modelstd$, identical to E2's.

    Built through the shared `advtr.clean_target_spec` with the same 50/50
    dataset-level split selector, so the spec (and therefore the content hash)
    matches E2's clean baseline exactly and the two share one checkpoint.

    Args:
        ctx: The run context (level, seed).
        dataset: The dataset name.
        num_features: Input feature count for the dense architectures.
        num_classes: Number of output classes.
        batch_size: Training batch size (paper batch, or the tiny batch at `test`).
        capacity: The capacity tier.

    Returns:
        The clean-baseline spec.
    """
    return advtr.clean_target_spec(
        ctx.level,
        dataset,
        ctx.seed,
        capacity,
        num_features,
        num_classes,
        batch_size,
        advtr.DATASET_SPLIT_TARGET,
    )


def retrain_after_outlier_removal(
    clean: nn.Module,
    target_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: str,
    epochs: int,
    batch_size: int,
    percent: int,
) -> nn.Module:
    """Purify the target's training data via kNN-Shapley and retrain.

    Mirrors the old `run_outlier_removal.py`: the *trained* clean target seeds
    the defense (its penultimate features drive the kNN-Shapley scores), the
    lowest-influence `percent`% of records are dropped, and the model is
    retrained on the remainder. The clean target is deep-copied first, so the
    shared $\\modelstd$ the baseline row measures is never mutated.

    Args:
        clean: The trained clean baseline to purify around. Not modified.
        target_loader: The target half's training data (unshuffled), scored for outliers.
        test_loader: The held-out test data the Shapley scores are computed against.
        device: Device to train and score on.
        epochs: Retraining epochs.
        batch_size: Retraining batch size.
        percent: Percentage of lowest-influence records to remove.

    Returns:
        The retrained defended model.
    """
    from amulet.poisoning.defenses import OutlierRemoval

    starting = copy.deepcopy(clean)
    defense = OutlierRemoval(
        starting,
        nn.CrossEntropyLoss(),
        torch.optim.Adam(starting.parameters(), lr=1e-3),
        target_loader,
        test_loader,
        device,
        percent=percent,
        epochs=epochs,
        batch_size=batch_size,
    )
    return defense.train_robust()


def build_models(
    ctx: advtr.RunContext, dataset: str, percent: int, capacity: str = CAPACITY
) -> tuple[ModelBundle, AmuletDataset]:
    """Train (or load) the clean, defended and stolen models for one cell.

    Args:
        ctx: The run context (level, seed, device, cache directory).
        dataset: The dataset name.
        percent: The removal percentage (the swept value, which keys the row).
        capacity: The capacity tier.

    Returns:
        The models with their specs, and the loaded dataset.
    """
    from amulet.unauth_model_ownership.attacks import ModelExtraction

    data = ctx.data(dataset)
    num_features, num_classes = data.num_features, data.num_classes
    batch_size = advtr.batch_for(ctx.level, BATCH_SIZE)
    epochs = advtr.epochs_for(ctx.level)

    target_set, adversary_set = advtr.dataset_adversary_split(data.train_set, ctx.seed)
    target_loader = advtr.loader_for(target_set, batch_size)
    adversary_loader = advtr.loader_for(adversary_set, batch_size)
    test_loader = advtr.loader_for(data.test_set, batch_size)

    # The clean baseline, shared with E2: same spec, same content hash, one file.
    clean_spec = clean_baseline_spec(
        ctx, dataset, num_features, num_classes, batch_size, capacity
    )
    clean = ctx.get_or_train(
        clean_spec,
        num_features,
        num_classes,
        lambda model: advtr.train_clean(
            model, target_loader, ctx.device, clean_spec.epochs
        ),
    )

    if percent == 0:
        # No outliers removed: the defended model *is* the clean baseline. This is
        # the table's $\\modelstd$ column / the figures' leftmost point.
        defended, defended_spec = clean, clean_spec
    else:
        defended_spec = clean_spec.replace(optimizer_recipe=outrem_recipe(percent))
        defended = ctx.get_or_train(
            defended_spec,
            num_features,
            num_classes,
            lambda _model: retrain_after_outlier_removal(
                clean,
                target_loader,
                test_loader,
                ctx.device,
                epochs,
                batch_size,
                percent,
            ),
        )

    stolen_spec = clean_spec.replace(
        optimizer_recipe=stolen_recipe(percent),
        subset_selector=advtr.DATASET_SPLIT_ADVERSARY,
    )

    def distil(model: nn.Module) -> nn.Module:
        # The surrogate is distilled from the DEFENDED model (which is the clean
        # baseline at percent 0), querying it with the adversary's held-out half.
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
        ModelBundle(clean, defended, stolen, clean_spec, defended_spec, stolen_spec),
        data,
    )


def run_cell(
    ctx: advtr.RunContext, dataset: str, percent: int, output_dir: Path
) -> list[dict[str, object]]:
    """Run one (dataset, seed, percent) cell and append its result row.

    Args:
        ctx: The run context.
        dataset: The dataset name.
        percent: The swept removal percentage.
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
        "percent": percent,
    }
    if row_exists(output, SCHEMA, key):
        return []

    bundle, data = build_models(ctx, dataset, percent)
    batch_size = advtr.batch_for(ctx.level, BATCH_SIZE)
    test_loader = advtr.loader_for(data.test_set, batch_size)

    # `evaluate_extraction` scores the surrogate against `defended` as the
    # reference, so `target_accuracy` here is the defended model's test accuracy
    # (the clean baseline's at percent 0).
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
        "percent": percent,
        "defended_test_acc": scores["target_accuracy"],
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
    percents: tuple[int, ...] = PERCENTS,
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    device: str | None = None,
) -> list[dict[str, object]]:
    """Run E4 at one verification level and return the rows it appended.

    Args:
        level: One of `common.config.LEVEL_NAMES`.
        seeds: Seeds to sweep. None keeps the level's own seeds.
        datasets: Datasets to sweep, a subset of `schemas.DATASETS`.
        percents: Removal percentages to sweep, a subset of `schemas.PERCENTS`.
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
            level=config,
            seed=seed,
            device=resolved_device,
            cache_dir=resolved_cache,
            tiny_data_factory=tiny_outrem_dataset,
        )
        for dataset in datasets:
            for percent in percents:
                rows.extend(run_cell(ctx, dataset, percent, directory))
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


def _parse_percents(text: str) -> tuple[int, ...]:
    """Parse a comma-separated removal-percentage subset, validated against `PERCENTS`."""
    if text.strip() == "all":
        return PERCENTS
    requested = [int(piece.strip()) for piece in text.split(",") if piece.strip()]
    unknown = [value for value in requested if value not in PERCENTS]
    if unknown:
        known = ", ".join(str(value) for value in PERCENTS)
        raise ValueError(
            f"Unknown percent(s): {', '.join(str(v) for v in unknown)}. Choose from: {known}."
        )
    return tuple(value for value in PERCENTS if value in set(requested))


def main(argv: list[str] | None = None) -> None:
    """Run E4 from the command line."""
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
        "--percents",
        type=str,
        default="all",
        help=f"Comma-separated subset of: {', '.join(str(v) for v in PERCENTS)}. Default: all.",
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
        percents=_parse_percents(args.percents),
        device=args.device,
    )


if __name__ == "__main__":
    main()
