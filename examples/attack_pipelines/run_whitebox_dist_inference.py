"""
End-to-end pipeline: run the White-Box PIM distribution inference attack.

Trains two populations of models (one per distribution ratio) on a tabular
dataset, then uses a Permutation Invariant Model (PIM) meta-classifier to
infer which distribution a held-out victim model was trained on based on
its raw parameters.
"""

import sys

sys.path.append("../../")

import argparse
import logging
from pathlib import Path

import torch

from amulet.distribution_inference.attacks import WhiteBoxPIM
from amulet.distribution_inference.metrics import evaluate_distinguishing_accuracy
from amulet.utils import create_dir, load_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../../")
    parser.add_argument(
        "--dataset", type=str, default="census", help="Options: census, lfw."
    )
    parser.add_argument("--model", type=str, default="linearnet")
    parser.add_argument("--model_capacity", type=str, default="m1")
    parser.add_argument("--training_size", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--meta_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--exp_id", type=int, default=0)
    parser.add_argument("--ratio1", type=float, default=0.1)
    parser.add_argument("--ratio2", type=float, default=0.9)
    parser.add_argument(
        "--filter_column",
        type=str,
        default="sex",
        help=(
            "Sensitive column whose proportion is being inferred. Must be one "
            "of the dataset's sensitive_columns. For census: 'sex' or 'race'. "
            "For lfw with default attributes: 'race' or 'gender'."
        ),
    )
    parser.add_argument(
        "--filter_value",
        type=int,
        default=1,
        help="Value of filter_column that satisfies the filter.",
    )
    parser.add_argument("--num_models", type=int, default=5)
    parser.add_argument("--train_subsample", type=int, default=1100)
    parser.add_argument("--test_subsample", type=int, default=500)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    root_dir = Path(args.root)
    log_dir = root_dir / "logs"
    create_dir(log_dir)
    logging.basicConfig(
        level=logging.INFO,
        filename=log_dir / "whitebox_dist_inference.log",
        filemode="w",
    )
    log = logging.getLogger("whitebox_dist_inference")
    log.addHandler(logging.StreamHandler())

    torch.manual_seed(args.exp_id)

    data = load_data(root_dir, args.dataset, args.training_size, log)
    if (
        data.x_train is None
        or data.y_train is None
        or data.x_test is None
        or data.y_test is None
        or data.z_train is None
        or data.z_test is None
        or data.sensitive_columns is None
    ):
        raise RuntimeError(
            f"Dataset '{args.dataset}' does not expose the arrays required for "
            "distribution inference (x_*, y_*, z_*, sensitive_columns)."
        )

    attack = WhiteBoxPIM(
        x_train=data.x_train,
        y_train=data.y_train,
        z_train=data.z_train,
        x_test=data.x_test,
        y_test=data.y_test,
        z_test=data.z_test,
        sensitive_columns=data.sensitive_columns,
        filter_column=args.filter_column,
        ratio1=args.ratio1,
        ratio2=args.ratio2,
        model_arch=args.model,
        model_capacity=args.model_capacity,
        num_features=data.num_features,
        num_classes=data.num_classes,
        num_models=args.num_models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        models_dir=root_dir / "models" / "distribution_inference",
        dataset=args.dataset,
        exp_id=args.exp_id,
        filter_value=args.filter_value,
        train_subsample=args.train_subsample,
        test_subsample=args.test_subsample,
        meta_epochs=args.meta_epochs,
        lr=args.lr,
    )

    attack.prepare_model_populations()
    results = attack.attack()

    metrics = evaluate_distinguishing_accuracy(
        results["predictions"], results["ground_truth"]
    )
    log.info("White-box PIM distribution inference results: %s", metrics)
    print(metrics)


if __name__ == "__main__":
    main(parse_args())
