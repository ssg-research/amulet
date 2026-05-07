import sys

sys.path.append("../../")

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from amulet.evasion.attacks import EvasionPGD
from amulet.evasion.defenses import AdversarialTrainingPGD
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
    parser.add_argument(
        "--epsilon", type=float, default=0.01, help="Controls amount of perturbations."
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Setup logger
    root_dir = Path(args.root)
    log_dir = root_dir / "logs"
    create_dir(log_dir)
    logging.basicConfig(
        level=logging.INFO, filename=log_dir / "adversarial_training.log", filemode="w"
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

    # Train or load model with Adversarial Training
    defended_model_filename = (
        models_path / "adversarial_training" / f"epsilon_{args.epsilon}" / filename
    )

    def _init_defended():
        return initialize_model(
            args.model, args.model_capacity, data.num_features, data.num_classes, log
        ).to(args.device)

    def _train_defended(_model):
        optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)
        adv_training = AdversarialTrainingPGD(
            target_model,
            criterion,
            optimizer,
            train_loader,
            args.device,
            args.epochs,
            args.epsilon,
        )
        return adv_training.train_robust()

    defended_model = load_or_train(
        defended_model_filename, _init_defended, _train_defended, log, "defended model"
    )

    test_accuracy_defended = get_accuracy(defended_model, test_loader, args.device)
    log.info("Test accuracy of defended model: %s", test_accuracy_defended)

    # Show the defense benefit: compare adversarial accuracy before and after
    evasion = EvasionPGD(
        target_model, test_loader, args.device, args.batch_size, args.epsilon
    )
    adv_loader = evasion.attack()
    adv_accuracy_target = get_accuracy(target_model, adv_loader, args.device)
    log.info("Adversarial accuracy (undefended): %s", adv_accuracy_target)

    evasion_def = EvasionPGD(
        defended_model, test_loader, args.device, args.batch_size, args.epsilon
    )
    adv_loader_def = evasion_def.attack()
    adv_accuracy_defended = get_accuracy(defended_model, adv_loader_def, args.device)
    log.info("Adversarial accuracy (defended): %s", adv_accuracy_defended)


if __name__ == "__main__":
    args = parse_args()
    main(args)
