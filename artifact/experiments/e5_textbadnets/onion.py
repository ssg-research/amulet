"""E5, intended interaction: ONION vs. the TextBadNets LLM backdoor.

Same-risk composition: ONION (a poisoning defense) meets a poisoning attack. Three
conditions per poison rate: a clean baseline, an undefended poisoned target, and an
ONION-defended target that trains on ONION-purified poisoned data and is evaluated on
ONION-purified inputs. Realizes H1 (attack) and H2 (intended interaction). See
experiments/text_backdoor_experiments.md in the repository root.

Per seed the clean baseline is trained once and the poison-rate grid is swept internally;
one row is appended per rate to `results/e5_textbadnets/onion.csv`. `exp_id` is the seed
everywhere.

    python artifact/experiments/e5_textbadnets/onion.py --level test
    python artifact/experiments/e5_textbadnets/onion.py --level full --seeds 0-4

Levels come from `common.config` (plan §8). `test` is the old `--smoke` path: a tiny
random-init target on CPU. `smoke` keeps the real Llama but reads a tenth of the corpus
for one epoch. `full` is the paper run. Requires the LLM extra: `uv sync --extra cu130
--extra llm` (or `--extra cpu` for `--level test`).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import tempfile
import time
from typing import TYPE_CHECKING, cast

import torch

from amulet.datasets import AmuletDataset, TextTensorDataset
from amulet.poisoning.attacks import TextBadNets
from amulet.poisoning.defenses import ONION
from common.config import LEVEL_NAMES, get_level
from common.io import append_row, row_exists, run_output_dir
from experiments.e5_textbadnets.llm_backdoor_common import (
    TargetFactory,
    accuracy,
    cached_purify,
    load_sst2_seeded,
    make_smoke_setup,
    make_target_factory,
    onion_stats,
    train_target,
)
from experiments.e5_textbadnets.schemas import ONION_SCHEMA

if TYPE_CHECKING:
    from common.config import LevelConfig

EXPERIMENT_ID = "e5_textbadnets"
CSV_STEM = "onion"

# The paper trains for three epochs on the full 67,349-record SST-2 train split. The
# `full` level leaves epochs unset by design (each experiment's count differs), so this
# is what fills it; `test` and `smoke` keep their own one-epoch budget.
PAPER_EPOCHS = 3
SST2_TRAIN_SIZE = 67349


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the study's knobs, level and seeds included."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=str, default="full", choices=LEVEL_NAMES)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Seeds to sweep, e.g. `0` or `0-4`. Default: the level's own seeds.",
    )
    parser.add_argument("--dataset", type=str, default="sst2", choices=["sst2"])
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument(
        "--reference_model",
        type=str,
        default=None,
        help="Reference LM recorded for ONION's perplexity scoring. Defaults to model_name: "
        "ONION scores through the target's own clean base LM, not an external model.",
    )
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--trigger", type=str, default="cf")
    parser.add_argument("--trigger_label", type=int, default=1)
    parser.add_argument(
        "--insert_position",
        type=str,
        default="random",
        choices=["start", "random", "end"],
    )
    parser.add_argument(
        "--poisoned_portions",
        type=str,
        default="0.0001,0.001,0.01,0.02,0.05",
        help="Comma-separated poison rates to sweep (attack strength).",
    )
    parser.add_argument(
        "--onion_threshold",
        type=float,
        default=0.0,
        help="ONION suspicion cutoff (single calibrated value; applied to train and test).",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="ONION purify cache dir. Defaults to the shared .onion_cache beside this "
        "script; set a per-seed dir (e.g. .onion_cache_3) to isolate parallel runs from "
        "cache write races.",
    )
    # Set by `apply_level` from the chosen level, declared here so the namespace is
    # complete however the run was started.
    parser.add_argument("--exp_id", type=int, default=0, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=PAPER_EPOCHS)
    parser.add_argument(
        "--max_train_samples", type=int, default=-1, help="-1 = full (67349)."
    )
    parser.add_argument(
        "--max_test_samples", type=int, default=-1, help="-1 = full (872)."
    )
    args = parser.parse_args(argv)
    if args.reference_model is None:
        args.reference_model = args.model_name
    return args


def run_experiment(
    args: argparse.Namespace,
    data: AmuletDataset,
    factory: TargetFactory,
    cache_dir: Path,
    output: Path,
) -> list[dict[str, object]]:
    """Sweep the poison-rate grid for one seed, appending a row per rate."""
    device = args.device
    dtype = "float32"
    portions = [float(p) for p in str(args.poisoned_portions).split(",")]
    train_set = cast(TextTensorDataset, data.train_set)
    test_set = cast(TextTensorDataset, data.test_set)

    # Clean baseline (condition 1) — poison-rate-independent, trained once per seed.
    clean_model, clean_runtime = train_target(
        factory,
        train_set,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        seed=args.exp_id,
    )
    clean_baseline_test_acc = accuracy(clean_model, test_set, device, args.batch_size)
    del clean_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # ONION scores perplexity with the target's own clean base LM (adapters off), so its
    # reference is a fresh unpoisoned target rather than an external GPT-2. Loaded once;
    # threshold applied to both train and test purification.
    from amulet.datasets.__text_datasets import _load_tokenizer

    onion = ONION(
        model=factory(),
        tokenizer=_load_tokenizer(train_set.tokenizer_name),
        threshold=args.onion_threshold,
        device=device,
    )
    # Attack-independent: purify the clean test once, reuse across every poison rate. The
    # cache key tracks the model_name (the true reference LM now) so old GPT-2-keyed
    # entries can't wrongly hit.
    purified_clean_test = cached_purify(onion, test_set, args.model_name, cache_dir)

    rows: list[dict[str, object]] = []
    for portion in portions:
        if row_exists(
            output, ONION_SCHEMA, {"exp_id": args.exp_id, "poisoned_portion": portion}
        ):
            print(f"skip exp_id={args.exp_id} p={portion}")
            continue

        attack = TextBadNets(
            trigger=args.trigger,
            trigger_label=args.trigger_label,
            portion=portion,
            random_seed=args.exp_id,
            insert_position=args.insert_position,
        )
        poisoned_train = attack.poison_train(train_set)
        poisoned_test = attack.poison_test(test_set)
        labels = [int(y) for y in train_set.tensors[1].tolist()]
        n_poisoned = len(attack.select_poison_indices(labels, len(labels)))

        # Condition 2 — undefended poisoned target.
        undef_model, undef_runtime = train_target(
            factory,
            poisoned_train,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            seed=args.exp_id,
        )
        undef_test_acc = accuracy(undef_model, test_set, device, args.batch_size)
        undef_asr = accuracy(undef_model, poisoned_test, device, args.batch_size)
        del undef_model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        # Condition 3 — ONION-defended: train on purified poison, eval on purified inputs.
        purify_start = time.perf_counter()
        purified_train = cached_purify(
            onion, poisoned_train, args.model_name, cache_dir
        )
        purified_triggered_test = cached_purify(
            onion, poisoned_test, args.model_name, cache_dir
        )
        onion_purify_runtime = time.perf_counter() - purify_start

        def_model, def_runtime = train_target(
            factory,
            purified_train,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            seed=args.exp_id,
        )
        def_test_acc_purified = accuracy(
            def_model, purified_clean_test, device, args.batch_size
        )
        def_test_acc_raw = accuracy(def_model, test_set, device, args.batch_size)
        def_asr = accuracy(def_model, purified_triggered_test, device, args.batch_size)
        removal_rate, mean_removed = onion_stats(
            args.trigger, list(poisoned_test.texts), list(purified_triggered_test.texts)
        )
        del def_model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        row: dict[str, object] = {
            "exp_id": args.exp_id,
            "dataset": args.dataset,
            "model_name": args.model_name,
            "reference_model": args.reference_model,
            "dtype": dtype,
            "num_classes": data.num_classes,
            "max_length": args.max_length,
            "n_train": len(train_set),
            "clean_test_size": len(test_set),
            "asr_test_size": len(poisoned_test),
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "trigger": args.trigger,
            "trigger_label": args.trigger_label,
            "insert_position": args.insert_position,
            "poisoned_portion": portion,
            "n_poisoned_train": n_poisoned,
            "onion_threshold": args.onion_threshold,
            "clean_baseline_test_acc": clean_baseline_test_acc,
            "undef_test_acc": undef_test_acc,
            "undef_asr": undef_asr,
            "def_test_acc_purified": def_test_acc_purified,
            "def_test_acc_raw": def_test_acc_raw,
            "def_asr": def_asr,
            "trigger_removal_rate": removal_rate,
            "mean_words_removed": mean_removed,
            "clean_train_runtime_sec": round(clean_runtime, 2),
            "undef_train_runtime_sec": round(undef_runtime, 2),
            "def_train_runtime_sec": round(def_runtime, 2),
            "onion_purify_runtime_sec": round(onion_purify_runtime, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _ = append_row(output, ONION_SCHEMA, row)
        rows.append(row)
        print(
            f"exp_id={args.exp_id} p={portion} | clean {clean_baseline_test_acc:.1f} | "
            f"undef acc {undef_test_acc:.1f} asr {undef_asr:.1f} | onion acc "
            f"{def_test_acc_purified:.1f} asr {def_asr:.1f} (removed {removal_rate:.2f})"
        )
    return rows


def apply_level(args: argparse.Namespace, config: LevelConfig, seed: int) -> None:
    """Point one seed's run at the budget its verification level allows.

    `test` is the old `--smoke` path: a tiny random-init target on eight synthetic
    sentences, so the poison rate rises to one half (eight sentences cannot carry a
    0.01% rate) and the device is forced to CPU. `smoke` and `full` keep the real target
    and the paper's poison-rate grid, differing in epochs and how much of SST-2 they read.

    Args:
        args: Namespace mutated in place.
        config: The level preset, already carrying `epochs`.
        seed: The seed, recorded as `exp_id`.
    """
    args.exp_id = seed
    args.epochs = config.epochs
    if config.tiny_model:
        args.device = "cpu"
        args.batch_size = 4
        args.poisoned_portions = "0.5"
        args.onion_threshold = 0.0
    elif config.train_fraction < 1.0:
        args.max_train_samples = round(config.train_fraction * SST2_TRAIN_SIZE)


def build_inputs(
    args: argparse.Namespace, config: LevelConfig
) -> tuple[AmuletDataset, TargetFactory]:
    """Build the dataset and the target factory this level calls for.

    Args:
        args: The parsed knobs, already levelled.
        config: The level preset.

    Returns:
        The dataset and a factory producing a fresh untrained target.
    """
    if config.tiny_model:
        return make_smoke_setup()
    data = load_sst2_seeded(
        args.exp_id,
        args.model_name,
        args.max_length,
        None if args.max_train_samples < 0 else args.max_train_samples,
        None if args.max_test_samples < 0 else args.max_test_samples,
    )
    factory = make_target_factory(
        args.model_name,
        data.num_classes,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        args.device,
    )
    return data, factory


def _cache_dir(args: argparse.Namespace, config: LevelConfig) -> Path:
    """Return the ONION purification cache this run should use."""
    if args.cache_dir is not None:
        return Path(args.cache_dir)
    if config.tiny_model:
        # A tiny random-init target's perplexities have nothing to do with the paper
        # run's, and `--level test` should leave no trace in the shared cache.
        return Path(tempfile.mkdtemp(prefix="e5_onion_test_cache_"))
    from experiments.e5_textbadnets.llm_backdoor_common import _DEFAULT_ONION_CACHE

    return _DEFAULT_ONION_CACHE


def default_output_dir(config: LevelConfig) -> Path:
    """Return the directory this level's result CSV belongs in.

    Every level writes under `runs/<level>/<experiment_id>/`, never into the
    committed `results/` tree: a `full` re-run must not clobber the shipped
    ground truth, and a `smoke`/`test` run must not have its reduced-budget or
    random-init numbers averaged into the paper's. The `runs/<level>/` tree
    mirrors `results/` (E5's `onion.csv`/`dp.csv` live in a `<experiment_id>/`
    subdirectory of both), so a `make_*` renderer reads either the same way.
    Authors promote a completed `full` run by copying its CSVs into `results/`.

    Args:
        config: The level preset.

    Returns:
        An existing or creatable directory.
    """
    return run_output_dir(config.name) / EXPERIMENT_ID


def run_level(
    config: LevelConfig,
    seed: int,
    output_dir: Path | None = None,
    argv: list[str] | None = None,
) -> list[dict[str, object]]:
    """Run one seed of the ONION study at one verification level.

    Args:
        config: The level preset, already carrying `epochs`.
        seed: The seed, recorded as `exp_id` and used for every RNG.
        output_dir: Directory the result CSV goes in. Defaults per level.
        argv: Command-line overrides for the study's own knobs.

    Returns:
        The rows this call appended. Cells already recorded are skipped, so a resumed
        sweep returns only what it added.
    """
    args = parse_args([] if argv is None else argv)
    apply_level(args, config, seed)
    data, factory = build_inputs(args, config)
    directory = output_dir if output_dir is not None else default_output_dir(config)
    return run_experiment(
        args, data, factory, _cache_dir(args, config), directory / f"{CSV_STEM}.csv"
    )


def main(argv: list[str] | None = None) -> None:
    """Run the ONION study over the level's seeds from the command line."""
    from experiments.e5_textbadnets.run import parse_seeds

    arguments = sys.argv[1:] if argv is None else argv
    args = parse_args(arguments)
    config = get_level(args.level).with_defaults(epochs=PAPER_EPOCHS)
    if args.seeds is not None:
        config = config.override(seeds=parse_seeds(args.seeds))
    for seed in config.seeds:
        _ = run_level(config, seed, argv=arguments)


if __name__ == "__main__":
    main()
