"""Textual backdoor poisoning on a LoRA-tuned decoder LLM, with two defenses.

This is the runnable efficacy surface for the LLM backdoor work and doubles as the
"how to extend Amulet to an LLM" example. It flows the standard pipeline shape
(AmuletDataset -> attack -> train -> metric) with a genuine decoder LLM victim:

  1. Load a text dataset (SST-2 / AG News / IMDB) as token-id tensors.
  2. TextBadNets stamps a trigger into a fraction of training rows and flips their
     labels; poison_test triggers every non-target test row (its accuracy is ASR).
  3. Fine-tune HFTextClassifier (AutoModelForSequenceClassification + LoRA) on the
     poisoned data and report clean accuracy and ASR.
  4. Report two interactions:
       - intended:   ONION purifies the triggered inputs, then re-measure ASR.
       - unintended: DP-LoRA fine-tuning (reused DPSGD) bounds each example's
                     influence; re-measure ASR and report epsilon.

Requires the optional LLM stack: `uv sync --extra llm`.

Example:
    uv run python run_text_backdoor.py --dataset sst2 --defense both \\
        --max_train_samples 2000 --max_test_samples 500 --epochs 3
"""

import sys

sys.path.append("../../")

import argparse
import logging
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader

from amulet.datasets import (
    AmuletDataset,
    TextTensorDataset,
    load_agnews,
    load_imdb,
    load_sst2,
)
from amulet.membership_inference.defenses import DPSGD
from amulet.models import HFTextClassifier
from amulet.poisoning.attacks import TextBadNets
from amulet.poisoning.defenses import ONION
from amulet.utils import create_dir, get_accuracy, train_classifier

_LOADERS = {"sst2": load_sst2, "agnews": load_agnews, "imdb": load_imdb}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="../../",
        help="Root directory of models and datasets.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sst2",
        choices=list(_LOADERS),
        help="Text dataset: sst2, agnews, or imdb.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base decoder LLM (Hub id). Swap for meta-llama/Llama-3.2-1B with access.",
    )
    parser.add_argument(
        "--max_length", type=int, default=128, help="Fixed token sequence length."
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=2000,
        help="Cap on training rows (keeps the demo tractable). Use -1 for all.",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=500,
        help="Cap on test rows. Use -1 for all.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size of input data."
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs for LoRA fine-tuning."
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="LoRA learning rate.")
    parser.add_argument(
        "--poisoned_portion",
        type=float,
        default=0.1,
        help="Fraction of training rows to poison (range 0 to 1).",
    )
    parser.add_argument(
        "--trigger", type=str, default="cf", help="Rare-word or phrase trigger."
    )
    parser.add_argument(
        "--trigger_label", type=int, default=1, help="Target label for poisoned rows."
    )
    parser.add_argument(
        "--insert_position",
        type=str,
        default="start",
        choices=["start", "random", "end"],
        help="Where the trigger is inserted in the text.",
    )
    parser.add_argument(
        "--defense",
        type=str,
        default="both",
        choices=["none", "onion", "dp", "both"],
        help="Which defense(s) to evaluate against the backdoor.",
    )
    parser.add_argument(
        "--onion_threshold",
        type=float,
        default=0.0,
        help="ONION suspicion cutoff; words scoring above it are removed.",
    )
    parser.add_argument(
        "--reference_model",
        type=str,
        default="gpt2",
        help="Reference LM ONION uses to score perplexity.",
    )
    parser.add_argument("--sigma", type=float, default=1.0, help="DP noise multiplier.")
    parser.add_argument("--delta", type=float, default=1e-5, help="DP target delta.")
    parser.add_argument(
        "--max_per_sample_grad_norm",
        type=float,
        default=1.0,
        help="DP per-sample gradient clipping norm.",
    )
    parser.add_argument(
        "--dp_epochs", type=int, default=3, help="Epochs for DP-LoRA fine-tuning."
    )
    parser.add_argument(
        "--dp_lr", type=float, default=1e-3, help="DP-LoRA learning rate."
    )
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device on which to run PyTorch.",
    )
    parser.add_argument(
        "--exp_id", type=int, default=0, help="Used as a random seed for experiments."
    )

    return parser.parse_args()


def load_text_dataset(args: argparse.Namespace) -> AmuletDataset:
    """Load the requested text dataset, tokenized with the victim's tokenizer."""
    max_train = None if args.max_train_samples < 0 else args.max_train_samples
    max_test = None if args.max_test_samples < 0 else args.max_test_samples
    data_path = Path(args.root) / "data" / args.dataset
    return _LOADERS[args.dataset](
        path=data_path,
        tokenizer_name=args.model_name,
        max_length=args.max_length,
        max_train_samples=max_train,
        max_test_samples=max_test,
    )


def build_victim(args: argparse.Namespace, num_classes: int, dtype: torch.dtype):
    """Construct a fresh LoRA sequence-classifier victim on the target device."""
    return HFTextClassifier(
        model_name=args.model_name,
        num_labels=num_classes,
        lora_r=args.lora_r,
        dtype=dtype,
    ).to(args.device)


def main(args: argparse.Namespace) -> None:
    root_dir = Path(args.root)
    log_dir = root_dir / "logs"
    create_dir(log_dir)
    logging.basicConfig(
        level=logging.INFO, filename=log_dir / "text_backdoor.log", filemode="w"
    )
    log = logging.getLogger("All")
    log.addHandler(logging.StreamHandler())
    # The Hugging Face stack configures the root logger on import, which turns the
    # basicConfig above into a no-op; set the level explicitly so the result lines
    # below still surface to the console.
    log.setLevel(logging.INFO)

    torch.manual_seed(args.exp_id)

    # Load and poison. poison_train relabels a fraction of triggered rows; poison_test
    # triggers every non-target row so that accuracy on it against trigger_label is ASR.
    data = load_text_dataset(args)
    attack = TextBadNets(
        trigger=args.trigger,
        trigger_label=args.trigger_label,
        portion=args.poisoned_portion,
        random_seed=args.exp_id,
        insert_position=args.insert_position,
    )
    # For a text dataset the train/test sets are TextTensorDataset instances (carrying
    # raw `.texts`); the static type is the broader Dataset, so narrow it here.
    train_set = cast(TextTensorDataset, data.train_set)
    test_set = cast(TextTensorDataset, data.test_set)
    poisoned_train = attack.poison_train(train_set)
    poisoned_test = attack.poison_test(test_set)

    train_loader = DataLoader(poisoned_train, batch_size=args.batch_size, shuffle=True)
    clean_test_loader = DataLoader(data.test_set, batch_size=args.batch_size)
    asr_loader = DataLoader(poisoned_test, batch_size=args.batch_size)

    # Undefended backdoor: fine-tune the LoRA victim on the poisoned data.
    criterion = torch.nn.CrossEntropyLoss()
    victim = build_victim(args, data.num_classes, torch.bfloat16)
    optimizer = torch.optim.Adam(victim.trainable_parameters(), lr=args.lr)
    victim = train_classifier(
        victim, train_loader, criterion, optimizer, args.epochs, args.device
    )

    clean_acc = get_accuracy(victim, clean_test_loader, args.device)
    asr = get_accuracy(victim, asr_loader, args.device)
    log.info("Undefended | clean accuracy: %.2f | ASR: %.2f", clean_acc, asr)

    # Intended interaction: ONION purifies the triggered inputs before classification.
    if args.defense in ("onion", "both"):
        onion = ONION(
            reference_model_name=args.reference_model,
            threshold=args.onion_threshold,
            device=args.device,
        )
        purified_test = onion.purify(poisoned_test)
        purified_loader = DataLoader(purified_test, batch_size=args.batch_size)
        onion_asr = get_accuracy(victim, purified_loader, args.device)
        log.info(
            "ONION | ASR %.2f -> %.2f (clean accuracy unchanged: victim is the same)",
            asr,
            onion_asr,
        )

    # Unintended cross-risk interaction: DP-LoRA fine-tuning (reused DPSGD). The DP
    # victim must be fp32 (bf16 trainable params break Opacus per-sample clipping).
    if args.defense in ("dp", "both"):
        dp_victim = build_victim(args, data.num_classes, torch.float32)
        dp_optimizer = torch.optim.Adam(dp_victim.trainable_parameters(), lr=args.dp_lr)
        dp_training = DPSGD(
            model=dp_victim,
            criterion=criterion,
            optimizer=dp_optimizer,
            train_loader=train_loader,
            device=args.device,
            delta=args.delta,
            max_per_sample_grad_norm=args.max_per_sample_grad_norm,
            sigma=args.sigma,
            epochs=args.dp_epochs,
        )
        dp_model = dp_training.train_private()
        epsilon = dp_training.privacy_engine.accountant.get_epsilon(delta=args.delta)
        dp_clean_acc = get_accuracy(dp_model, clean_test_loader, args.device)
        dp_asr = get_accuracy(dp_model, asr_loader, args.device)
        log.info(
            "DP-LoRA (eps=%.2f) | clean accuracy: %.2f | ASR %.2f -> %.2f",
            epsilon,
            dp_clean_acc,
            asr,
            dp_asr,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
