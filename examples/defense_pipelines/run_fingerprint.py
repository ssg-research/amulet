import sys

sys.path.append("../../")

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from amulet.unauth_model_ownership.defenses import DatasetInference
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
        help="Options: cifar10, cifar100, fmnist, mnist, lfw, census, celeba, utkface.",
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

    parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument(
        "--randomize",
        help="For the individual attacks",
        type=int,
        default=0,
        choices=[0, 1, 2],
    )
    parser.add_argument(
        "--alpha_l1", help="Step Size for L1 attacks", type=float, default=1.0
    )
    parser.add_argument(
        "--alpha_l2", help="Step Size for L2 attacks", type=float, default=0.01
    )
    parser.add_argument(
        "--alpha_linf", help="Step Size for Linf attacks", type=float, default=0.001
    )
    parser.add_argument("--gap", help="For L1 attack", type=float, default=0.001)
    parser.add_argument("--k", help="For L1 attack", type=int, default=1)
    parser.add_argument(
        "--regressor_embed",
        help="Victim Embeddings for training regressor",
        type=int,
        default=0,
        choices=[0, 1],
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Setup logger
    root_dir = Path(args.root)
    log_dir = root_dir / "logs"
    create_dir(log_dir)
    logging.basicConfig(
        level=logging.INFO, filename=log_dir / "fingerprinting.log", filemode="w"
    )
    log = logging.getLogger("All")
    log.addHandler(logging.StreamHandler())

    # Set random seeds for reproducibility
    torch.manual_seed(args.exp_id)

    # Load dataset and create data loaders
    data = load_data(root_dir, args.dataset, args.training_size, log)
    train_loader = DataLoader(
        dataset=data.train_set, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        dataset=data.test_set, batch_size=args.batch_size, shuffle=False
    )

    # Set up filename and directories to save/load models
    models_path = root_dir / "saved_models"
    filename = f"{args.dataset}_{args.model}_{args.model_capacity}_{args.training_size * 100}_{args.batch_size}_{args.epochs}_{args.exp_id}.pt"
    target_model_filename = models_path / "target" / filename
    suspect_model_filename = models_path / "suspect" / filename

    # Train or Load Target and Suspect Models
    criterion = torch.nn.CrossEntropyLoss()

    def _init_model():
        return initialize_model(
            args.model, args.model_capacity, data.num_features, data.num_classes, log
        ).to(args.device)

    def _train_model(model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        return train_classifier(
            model, train_loader, criterion, optimizer, args.epochs, args.device
        )

    target_model = load_or_train(
        target_model_filename, _init_model, _train_model, log, "target model"
    )
    suspect_model = load_or_train(
        suspect_model_filename, _init_model, _train_model, log, "suspect model"
    )

    test_accuracy_target = get_accuracy(target_model, test_loader, args.device)
    log.info("Test accuracy of target model: %s", test_accuracy_target)

    # Run Fingerprinting
    log.info("Running Dataset Inference")
    dataset_format = "1D" if data.modality == "tabular" else "2D"

    dataset_inference = DatasetInference(
        target_model,
        suspect_model,
        train_loader,
        test_loader,
        data.num_classes,
        args.device,
        dataset_format,
        args.alpha_l1,
        args.alpha_l2,
        args.alpha_linf,
        args.k,
        args.gap,
        args.num_iter,
        args.regressor_embed,
        args.batch_size,
    )
    results = dataset_inference.fingerprint()
    log.info(results)


if __name__ == "__main__":
    args = parse_args()
    main(args)
