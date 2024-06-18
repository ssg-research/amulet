import sys

sys.path.append("../../")
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from amulet.unauth_model_ownership.defenses import Fingerprinting
from amulet.utils import (
    load_data,
    initialize_model,
    train_classifier,
    create_dir,
    get_accuracy,
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
        default="cifar10",
        help="Options: cifar10, fmnist, lfw, census.",
    )
    parser.add_argument(
        "--model", type=str, default="vgg", help="Options: vgg, linearnet, binarynet."
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
        default=torch.device(
            "cuda:{0}".format(0) if torch.cuda.is_available() else "cpu"
        ),
        help="Device on which to run PyTorch",
    )
    parser.add_argument(
        "--exp_id", type=int, default=0, help="Used as a random seed for experiments."
    )

    parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument(
        "--distance", help="Type of Adversarial Perturbation", type=str
    )  # , choices = ["linf", "l1", "l2", "vanilla"])
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
    generator = torch.Generator().manual_seed(args.exp_id)

    # Load dataset and create data loaders
    data = load_data(root_dir, generator, args.dataset, args.training_size, log)
    train_loader = DataLoader(
        dataset=data.train_set, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        dataset=data.test_set, batch_size=args.batch_size, shuffle=False
    )

    # Set up filename and directories to save/load models
    models_path = root_dir / "saved_models"
    filename = f"{args.dataset}_{args.model}_{args.model_capacity}_{args.training_size*100}_{args.batch_size}_{args.epochs}_{args.exp_id}.pt"
    target_model_path = models_path / "target"
    target_model_filename = target_model_path / filename

    # Train or Load Target Model
    criterion = torch.nn.CrossEntropyLoss()

    if target_model_filename.exists():
        log.info("Target model loaded from %s", target_model_filename)
        target_model = torch.load(target_model_filename).to(args.device)
        optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)
    else:
        log.info("Training target model")
        target_model = initialize_model(
            args.model, args.model_capacity, args.dataset, log
        ).to(args.device)
        optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)
        target_model = train_classifier(
            target_model, train_loader, criterion, optimizer, args.epochs, args.device
        )
        log.info("Target model trained")

        # Save model
        create_dir(target_model_path, log)
        torch.save(target_model, target_model_filename)

    test_accuracy_target = get_accuracy(target_model, test_loader, args.device)
    log.info("Test accuracy of target model: %s", test_accuracy_target)

    # Train or Load Suspect Model
    suspect_model_path = models_path / "suspect"
    suspect_model_filename = suspect_model_path / filename

    if suspect_model_filename.exists():
        log.info("Target model loaded from %s", suspect_model_filename)
        suspect_model = torch.load(suspect_model_filename).to(args.device)
        optimizer = torch.optim.Adam(suspect_model.parameters(), lr=1e-3)
    else:
        log.info("Training suspect model")
        suspect_model = initialize_model(
            args.model, args.model_capacity, args.dataset, log
        ).to(args.device)
        optimizer = torch.optim.Adam(suspect_model.parameters(), lr=1e-3)
        suspect_model = train_classifier(
            suspect_model, train_loader, criterion, optimizer, args.epochs, args.device
        )
        log.info("Target model trained")

        # Save model
        create_dir(suspect_model_path, log)
        torch.save(suspect_model, suspect_model_filename)

    # Run Fingerprinting
    log.info("Running Dataset Inference")
    num_classes_map = {"cifar10": 10, "fmnist": 10, "census": 2, "lfw": 2}
    dataset_map = {"cifar10": "2D", "fmnist": "2D", "census": "1D", "lfw": "1D"}

    fingerprinting = Fingerprinting(
        target_model,
        suspect_model,
        train_loader,
        test_loader,
        num_classes_map[args.dataset],
        args.device,
        args.distance,
        dataset_map[args.dataset],
        args.alpha_l1,
        args.alpha_l2,
        args.alpha_linf,
        args.k,
        args.gap,
        args.num_iter,
        args.regressor_embed,
        args.batch_size,
    )
    results = fingerprinting.dataset_inference()
    log.info(results)


if __name__ == "__main__":
    args = parse_args()
    main(args)
