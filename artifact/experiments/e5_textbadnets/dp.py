"""E5, unintended interaction: DP-SGD vs. the TextBadNets LLM backdoor.

Cross-risk composition: DP-SGD (a membership-inference/privacy defense, reused) applied
while measuring a poisoning attack. Three conditions per cell: a clean baseline, an
undefended poisoned target, and a DP-defended target trained on the same poisoned data
with per-example clipping and noise. The defense acts at training time only — test inputs
are unprocessed. Realizes H1 (attack) and H3 (unintended interaction). See
experiments/text_backdoor_experiments.md in the repository root.

Per seed the clean baseline is trained once and the poison-rate x target-epsilon grid is
swept internally (the undefended target is trained once per poison rate and reused across
that rate's epsilon rows); one row per cell lands in `results/e5_textbadnets/dp.csv`.
`exp_id` is the seed everywhere.

    python artifact/experiments/e5_textbadnets/dp.py --level test
    python artifact/experiments/e5_textbadnets/dp.py --level full --seeds 0-4

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
from opacus.accountants.utils import get_noise_multiplier
from torch.utils.data import DataLoader

from amulet.datasets import AmuletDataset, TextTensorDataset
from amulet.membership_inference.defenses import DPSGD
from amulet.poisoning.attacks import TextBadNets
from common.config import LEVEL_NAMES, get_level
from common.io import append_row, row_exists, run_output_dir
from experiments.e5_textbadnets.llm_backdoor_common import (
    DEFAULT_MODEL_CACHE,
    TargetFactory,
    accuracy,
    ckpt_key,
    get_or_train,
    load_sst2_seeded,
    make_smoke_setup,
    make_target_factory,
    seed_all,
    train_target,
)
from experiments.e5_textbadnets.schemas import DP_SCHEMA

if TYPE_CHECKING:
    from common.config import LevelConfig

EXPERIMENT_ID = "e5_textbadnets"
CSV_STEM = "dp"

# Match the accountant amulet's DPSGD PrivacyEngine uses, so the sigma we calibrate to a
# target epsilon and the epsilon the engine later reports agree.
_ACCOUNTANT = "prv"

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
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dp_epochs", type=int, default=None, help="Default: epochs.")
    parser.add_argument("--dp_lr", type=float, default=None, help="Default: lr.")
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
        default="0.001,0.01,0.05",
        help="Comma-separated poison rates to sweep (attack strength).",
    )
    parser.add_argument(
        "--target_epsilons",
        type=str,
        default="1.0,8.0",
        help="Comma-separated target epsilons to sweep.",
    )
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--max_per_sample_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--max_physical_batch_size",
        type=int,
        default=None,
        help="Cap physical micro-batch size (Opacus BatchMemoryManager) to bound "
        "per-sample-gradient memory. Default None = no splitting; ε is unaffected.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Shared-model checkpoint cache. The clean baseline (once/seed) and undefended "
        "target (once/rate) are trained once, cached here, and reused by every run that "
        "needs them, so no shared model is ever trained twice.",
    )
    parser.add_argument(
        "--clean_only",
        action="store_true",
        help="Train and cache only this seed's clean baseline, then exit. This is the "
        "per-seed gate to run before fanning out the rate configs, so the shared baseline "
        "exists before any of them start (they never race to train it).",
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
    return parser.parse_args(argv)


def train_dp(
    args: argparse.Namespace,
    factory: TargetFactory,
    poisoned_train: TextTensorDataset,
    target_epsilon: float,
    dp_lr: float,
    dp_epochs: int,
) -> tuple[torch.nn.Module, float, float, float]:
    """DP-SGD target calibrated to `target_epsilon`. Returns model, eps, sigma, runtime."""
    sample_rate = args.batch_size / len(poisoned_train)
    sigma = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=args.delta,
        sample_rate=sample_rate,
        epochs=dp_epochs,
        accountant=_ACCOUNTANT,
    )
    seed_all(args.exp_id)
    loader = DataLoader(poisoned_train, batch_size=args.batch_size)
    model = factory()
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=dp_lr)
    criterion = torch.nn.CrossEntropyLoss()
    dp = DPSGD(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=loader,
        device=args.device,
        delta=args.delta,
        max_per_sample_grad_norm=args.max_per_sample_grad_norm,
        sigma=sigma,
        epochs=dp_epochs,
        max_physical_batch_size=args.max_physical_batch_size,
    )
    start = time.perf_counter()
    dp_model = dp.train_private()
    runtime = time.perf_counter() - start
    epsilon = dp.privacy_engine.accountant.get_epsilon(delta=args.delta)
    return dp_model, epsilon, sigma, runtime


def run_experiment(
    args: argparse.Namespace,
    data: AmuletDataset,
    factory: TargetFactory,
    cache_dir: Path,
    output: Path,
) -> list[dict[str, object]]:
    """Sweep the poison-rate x epsilon grid for one seed, appending a row per cell."""
    device = args.device
    portions = [float(p) for p in str(args.poisoned_portions).split(",")]
    target_epsilons = [float(e) for e in str(args.target_epsilons).split(",")]
    dp_lr = args.dp_lr if args.dp_lr is not None else args.lr
    dp_epochs = args.dp_epochs if args.dp_epochs is not None else args.epochs
    train_set = cast(TextTensorDataset, data.train_set)
    test_set = cast(TextTensorDataset, data.test_set)

    # Hyperparams that identify a trained model, so a cached checkpoint is only reused by a
    # run whose model would be bit-for-bit the same. The clean baseline depends on all of
    # these; the undefended target additionally on the attack (poison rate + trigger).
    base_spec: dict[str, object] = {
        "seed": args.exp_id,
        "dataset": args.dataset,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "max_train": args.max_train_samples,
        "max_test": args.max_test_samples,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }
    clean_key = ckpt_key("clean", {**base_spec, "role": "clean"})

    def _train_clean() -> tuple[torch.nn.Module, dict[str, float]]:
        model, runtime = train_target(
            factory,
            train_set,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            seed=args.exp_id,
        )
        acc = accuracy(model, test_set, device, args.batch_size)
        return model, {
            "clean_baseline_test_acc": acc,
            "clean_train_runtime_sec": round(runtime, 2),
        }

    # Clean-only gate: train + cache this seed's shared baseline, then exit. Every rate
    # config for the seed then loads it from cache instead of retraining it. The payload
    # is the cached checkpoint under cache_dir; nothing is written to the result CSV,
    # which holds measurements rather than progress.
    if args.clean_only:
        m = get_or_train(cache_dir, clean_key, _train_clean)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(
            f"clean baseline ready exp_id={args.exp_id}: "
            f"test_acc={m['clean_baseline_test_acc']:.2f} "
            f"(runtime {m['clean_train_runtime_sec']:.0f}s)"
        )
        return []

    # Pending (portion -> epsilons not yet in the CSV). Resumable at cell granularity: a fully
    # done seed does no work (not even the clean baseline), and a half-done portion trains only
    # its missing epsilons. Adding 3.0 to --target_epsilons later re-enters and fills only the
    # e=3 cells, leaving the e=1/e=8 rows untouched.
    pending: dict[float, list[float]] = {}
    for portion in portions:
        todo = [
            e
            for e in target_epsilons
            if not row_exists(
                output,
                DP_SCHEMA,
                {
                    "exp_id": args.exp_id,
                    "poisoned_portion": portion,
                    "target_epsilon": e,
                },
            )
        ]
        if todo:
            pending[portion] = todo
    if not pending:
        print(f"skip exp_id={args.exp_id}: all poison x epsilon cells already done")
        return []

    # Clean baseline (condition 1) — poison-independent, trained once per seed and shared
    # across every cell. Loaded from cache (populated by the clean-only gate) on a hit.
    clean_metrics = get_or_train(cache_dir, clean_key, _train_clean)
    clean_baseline_test_acc = clean_metrics["clean_baseline_test_acc"]
    clean_runtime = clean_metrics["clean_train_runtime_sec"]
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    rows: list[dict[str, object]] = []
    for portion in portions:
        if portion not in pending:
            print(f"skip exp_id={args.exp_id} p={portion}: all epsilons already done")
            continue

        # Poison at this rate.
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

        # Undefended (condition 2) — trained once per rate, referenced in every epsilon row
        # and cached, so a config interrupted mid-epsilon and restarted reloads it (with the
        # clean baseline) instead of retraining ~5h of shared models before the DP target.
        undef_key = ckpt_key(
            "undef",
            {
                **base_spec,
                "role": "undef",
                "portion": portion,
                "trigger": args.trigger,
                "trigger_label": args.trigger_label,
                "insert_position": args.insert_position,
            },
        )

        def _train_undef(
            poisoned_train: TextTensorDataset = poisoned_train,
            poisoned_test: TextTensorDataset = poisoned_test,
        ) -> tuple[torch.nn.Module, dict[str, float]]:
            model, runtime = train_target(
                factory,
                poisoned_train,
                lr=args.lr,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                seed=args.exp_id,
            )
            return model, {
                "undef_test_acc": accuracy(model, test_set, device, args.batch_size),
                "undef_asr": accuracy(model, poisoned_test, device, args.batch_size),
                "undef_train_runtime_sec": round(runtime, 2),
            }

        undef_metrics = get_or_train(cache_dir, undef_key, _train_undef)
        undef_test_acc = undef_metrics["undef_test_acc"]
        undef_asr = undef_metrics["undef_asr"]
        undef_runtime = undef_metrics["undef_train_runtime_sec"]
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        for target_epsilon in pending[portion]:
            # DP-defended (condition 3).
            dp_model, epsilon, sigma, dp_runtime = train_dp(
                args, factory, poisoned_train, target_epsilon, dp_lr, dp_epochs
            )
            dp_test_acc = accuracy(dp_model, test_set, device, args.batch_size)
            dp_asr = accuracy(dp_model, poisoned_test, device, args.batch_size)
            del dp_model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

            row: dict[str, object] = {
                "exp_id": args.exp_id,
                "dataset": args.dataset,
                "model_name": args.model_name,
                "dtype": "float32",
                "num_classes": data.num_classes,
                "max_length": args.max_length,
                "n_train": len(train_set),
                "clean_test_size": len(test_set),
                "asr_test_size": len(poisoned_test),
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "dp_epochs": dp_epochs,
                "dp_lr": dp_lr,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "trigger": args.trigger,
                "trigger_label": args.trigger_label,
                "insert_position": args.insert_position,
                "poisoned_portion": portion,
                "n_poisoned_train": n_poisoned,
                "clean_baseline_test_acc": clean_baseline_test_acc,
                "undef_test_acc": undef_test_acc,
                "undef_asr": undef_asr,
                "target_epsilon": target_epsilon,
                "epsilon": epsilon,
                "sigma": sigma,
                "delta": args.delta,
                "max_per_sample_grad_norm": args.max_per_sample_grad_norm,
                "dp_test_acc": dp_test_acc,
                "dp_asr": dp_asr,
                "clean_train_runtime_sec": round(clean_runtime, 2),
                "undef_train_runtime_sec": round(undef_runtime, 2),
                "dp_train_runtime_sec": round(dp_runtime, 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            _ = append_row(output, DP_SCHEMA, row)
            rows.append(row)
            print(
                f"exp_id={args.exp_id} p={portion} target_eps={target_epsilon} "
                f"(eps={epsilon:.2f} sigma={sigma:.3f}) | clean {clean_baseline_test_acc:.1f} | "
                f"undef acc {undef_test_acc:.1f} asr {undef_asr:.1f} | dp acc {dp_test_acc:.1f} "
                f"asr {dp_asr:.1f}"
            )
    return rows


def apply_level(args: argparse.Namespace, config: LevelConfig, seed: int) -> None:
    """Point one seed's run at the budget its verification level allows.

    `test` is the old `--smoke` path: a tiny random-init target on eight synthetic
    sentences, so the poison rate rises to one half and one epsilon is enough. `smoke`
    and `full` keep the real target and the paper's grid, differing in epochs and how
    much of SST-2 they read.

    Args:
        args: Namespace mutated in place.
        config: The level preset, already carrying `epochs`.
        seed: The seed, recorded as `exp_id`.
    """
    args.exp_id = seed
    args.epochs = config.epochs
    if config.tiny_model:
        args.device = "cpu"
        args.poisoned_portions = "0.5"
        args.target_epsilons = "8.0"
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
    """Return the shared-model checkpoint cache this run should use."""
    if args.cache_dir is not None:
        return Path(args.cache_dir)
    if config.tiny_model:
        # An isolated cache still exercises the shared-model path (a second run reloads
        # the first's clean and undefended checkpoints) without touching the real one.
        return Path(tempfile.mkdtemp(prefix="e5_dp_test_cache_"))
    return DEFAULT_MODEL_CACHE


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
    """Run one seed of the DP-SGD study at one verification level.

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
    if config.tiny_model:
        # batch == train size keeps Opacus' Poisson sample rate at 1.0 (no empty batches).
        args.batch_size = len(cast(TextTensorDataset, data.train_set))
    directory = output_dir if output_dir is not None else default_output_dir(config)
    return run_experiment(
        args, data, factory, _cache_dir(args, config), directory / f"{CSV_STEM}.csv"
    )


def main(argv: list[str] | None = None) -> None:
    """Run the DP-SGD study over the level's seeds from the command line."""
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
