import sys

sys.path.append("../../")

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from amulet.membership_inference.attacks import LiRA
from amulet.membership_inference.metrics import compute_mi_metrics
from amulet.utils import (
    create_dir,
    get_accuracy,
    initialize_model,
    load_data,
    load_or_train,
    train_classifier,
)


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
        default="celeba",
        help="Options: cifar10, fmnist, lfw, census, celeba.",
    )
    parser.add_argument(
        "--model", type=str, default="vgg", help="Options: vgg, linearnet."
    )
    parser.add_argument(
        "--model_capacity",
        type=str,
        default="m1",
        help="Size of the model to use. Options: m1, m2, m3, m4, where m1 is the smallest.",
    )
    parser.add_argument(
        "--training_size", type=float, default=1, help="Fraction of dataset to use."
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size of input data."
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs for training."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device on which to run PyTorch",
    )
    parser.add_argument(
        "--exp_id", type=int, default=0, help="Used as a random seed for experiments."
    )
    parser.add_argument(
        "--pkeep",
        type=float,
        default=0.5,
        help="Proportion of data to keep for training.",
    )
    parser.add_argument(
        "--num_shadow", type=int, default=64, help="Number of shadow models to train."
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Setup logger
    root_dir = Path(args.root)
    log_dir = root_dir / "logs"
    create_dir(log_dir)
    logging.basicConfig(
        level=logging.INFO, filename=log_dir / "membership_inference.log", filemode="w"
    )
    log = logging.getLogger("All")
    log.addHandler(logging.StreamHandler())

    # Set random seeds for reproducibility
    torch.manual_seed(args.exp_id)

    # Load dataset and create data loaders
    data = load_data(root_dir, args.dataset, args.training_size, log)

    dataset_size: int = len(data.train_set)  # type: ignore[reportArgumentType]

    keep = np.random.choice(
        dataset_size,
        size=int(args.pkeep * dataset_size),
        replace=False,
    )
    keep.sort()
    target_train_set = Subset(
        data.train_set, list(keep)
    )  # ndarray not considered a Sequence
    train_loader = DataLoader(
        dataset=target_train_set, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        dataset=data.test_set, batch_size=args.batch_size, shuffle=False
    )

    # Set up filename and directories to save/load models
    models_path = root_dir / "saved_models"
    filename = f"{args.dataset}_{args.model}_{args.model_capacity}_{args.training_size * 100}_{args.batch_size}_{args.epochs}_{args.exp_id}.pt"
    target_model_filename = models_path / "target" / filename

    # Train or Load Target Model
    criterion = torch.nn.CrossEntropyLoss()

    def _init_target():
        return initialize_model(
            args.model, args.model_capacity, data.num_features, data.num_classes, log
        ).to(args.device)

    def _train_target(model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        return train_classifier(
            model, train_loader, criterion, optimizer, args.epochs, args.device
        )

    target_model = load_or_train(
        target_model_filename, _init_target, _train_target, log, "target model"
    )

    test_accuracy_target = get_accuracy(target_model, test_loader, args.device)
    log.info("Test accuracy of target model: %s", test_accuracy_target)

    # Run Membership Inference attack

    shadow_model_dir = models_path / "membership_inference" / "shadow_models"
    create_dir(shadow_model_dir, log)

    mem_inf = LiRA(
        target_model,
        keep,
        args.model,
        args.model_capacity,
        data.train_set,
        args.dataset,
        data.num_features,
        data.num_classes,
        args.batch_size,
        args.pkeep,
        criterion,
        args.num_shadow,
        args.epochs,
        args.device,
        shadow_model_dir,
        args.exp_id,
    )

    results = mem_inf.attack()

    print(compute_mi_metrics(results["lira_online_preds"], results["true_labels"]))
    print(compute_mi_metrics(results["lira_offline_preds"], results["true_labels"]))


if __name__ == "__main__":
    args = parse_args()
    main(args)
