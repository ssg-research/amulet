import sys

sys.path.append("../../")

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from amulet.unauth_model_ownership.attacks import ModelExtraction
from amulet.unauth_model_ownership.metrics import evaluate_extraction
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
        "--adv_train_fraction",
        type=float,
        default=0.5,
        help="Fraction of trianing data used by the adversary.",
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Setup logger
    root_dir = Path(args.root)
    log_dir = root_dir / "logs"
    create_dir(log_dir)
    logging.basicConfig(
        level=logging.INFO, filename=log_dir / "model_extraction.log", filemode="w"
    )
    log = logging.getLogger("All")
    log.addHandler(logging.StreamHandler())

    # Set random seeds for reproducibility
    torch.manual_seed(args.exp_id)

    # Load dataset and split train data for adversary
    data = load_data(root_dir, args.dataset, args.training_size, log)

    adv_train_size = int(args.adv_train_fraction * len(data.train_set))  # type: ignore[reportArgumentType]
    target_train_size = len(data.train_set) - adv_train_size  # type: ignore[reportArgumentType]
    generator = torch.Generator().manual_seed(args.exp_id)
    target_train_set, adv_train_set = random_split(
        data.train_set, [target_train_size, adv_train_size], generator=generator
    )

    # Create data loaders
    adv_train_loader = DataLoader(
        dataset=adv_train_set, batch_size=args.batch_size, shuffle=False
    )
    target_train_loader = DataLoader(
        dataset=target_train_set, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        dataset=data.test_set, batch_size=args.batch_size, shuffle=False
    )

    # Set up filename and directories to save/load models
    models_path = root_dir / "saved_models"
    filename = f"{args.dataset}_{args.model}_{args.model_capacity}_{args.training_size * 100}_{args.batch_size}_{args.epochs}_{args.exp_id}.pt"

    # Train or Load Target Model
    target_model_filename = (
        models_path
        / "targetForExtraction"
        / f"adv_train_fraction_{args.adv_train_fraction}"
        / filename
    )
    criterion = torch.nn.CrossEntropyLoss()

    def _init_target():
        return initialize_model(
            args.model, args.model_capacity, data.num_features, data.num_classes, log
        ).to(args.device)

    def _train_target(model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        return train_classifier(
            model, target_train_loader, criterion, optimizer, args.epochs, args.device
        )

    target_model = load_or_train(
        target_model_filename, _init_target, _train_target, log, "target model"
    )

    test_accuracy_target = get_accuracy(target_model, test_loader, args.device)
    log.info("Test accuracy of target model: %s", test_accuracy_target)

    # Train or Load model for Model Extraction
    attack_model_filename = (
        models_path
        / "model_extraction"
        / f"adv_train_fraction_{args.adv_train_fraction}"
        / filename
    )

    def _init_attack():
        return initialize_model(
            args.model, args.model_capacity, data.num_features, data.num_classes, log
        ).to(args.device)

    def _train_attack(model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model_extraction = ModelExtraction(
            target_model, model, optimizer, adv_train_loader, args.device, args.epochs
        )
        return model_extraction.attack()

    attack_model = load_or_train(
        attack_model_filename, _init_attack, _train_attack, log, "attack model"
    )

    # Use evaluate_attack() as a static method since the model might be loaded from file
    evaluation_results = evaluate_extraction(
        target_model, attack_model, test_loader, args.device
    )

    log.info("Target Model accuracy: %s", evaluation_results["target_accuracy"])
    log.info("Stolen Model accuracy: %s", evaluation_results["stolen_accuracy"])
    log.info("Fidelity: %s", evaluation_results["fidelity"])
    log.info("Correct Fidelity: %s", evaluation_results["correct_fidelity"])


if __name__ == "__main__":
    args = parse_args()
    main(args)
