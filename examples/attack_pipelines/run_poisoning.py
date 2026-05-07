import sys

sys.path.append("../../")

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from amulet.poisoning.attacks import BadNets
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
        "--epochs", type=int, default=2, help="Number of epochs for training."
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
        "--poisoned_portion",
        type=float,
        default=0.1,
        help="posioning portion (float, range from 0 to 1, default: 0.1)",
    )
    parser.add_argument(
        "--trigger_label",
        type=int,
        default=0,
        help="The NO. of trigger label (int, range from 0 to 10, default: 0)",
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Setup logger
    root_dir = Path(args.root)
    log_dir = root_dir / "logs"
    create_dir(log_dir)
    logging.basicConfig(
        level=logging.INFO, filename=log_dir / "model_poisoning.log", filemode="w"
    )
    log = logging.getLogger("All")
    log.addHandler(logging.StreamHandler())

    # Set random seeds for reproducibility
    torch.manual_seed(args.exp_id)

    # Load dataset and create data loaders
    data = load_data(root_dir, args.dataset, args.training_size, log)
    target_train_loader = DataLoader(
        dataset=data.train_set, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        dataset=data.test_set, batch_size=args.batch_size, shuffle=False
    )

    # Set up filename and directories to save/load models
    models_path = root_dir / "saved_models"
    filename = f"{args.dataset}_{args.model}_{args.model_capacity}_{args.training_size * 100}_{args.batch_size}_{args.epochs}_{args.exp_id}.pt"

    # Train or Load Target Model
    target_model_filename = models_path / "target" / filename
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

    # Train or Load model for Model Poisoning
    poisoned_model_filename = (
        models_path
        / "model_poisoning"
        / f"poisoned_protion_{args.poisoned_portion}"
        / filename
    )

    poisoning = BadNets(
        args.trigger_label,
        args.poisoned_portion,
        args.exp_id,
        data.modality,
    )

    def _init_poisoned():
        model = initialize_model(
            args.model, args.model_capacity, data.num_features, data.num_classes, log
        ).to(args.device)
        model.load_state_dict(target_model.state_dict())
        return model

    def _train_poisoned(model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        poisoned_train_set = poisoning.poison_train(data.train_set)
        poisoned_train_loader = DataLoader(
            dataset=poisoned_train_set, batch_size=args.batch_size, shuffle=False
        )
        return train_classifier(
            model,
            poisoned_train_loader,
            criterion,
            optimizer,
            args.epochs,
            args.device,
        )

    poisoned_model = load_or_train(
        poisoned_model_filename, _init_poisoned, _train_poisoned, log, "poisoned model"
    )

    poisoned_test_set = poisoning.poison_test(data.test_set)
    poisoned_test_loader = DataLoader(
        dataset=poisoned_test_set, batch_size=args.batch_size, shuffle=False
    )

    log.info(
        "Target Model on Origin Data: %s",
        get_accuracy(target_model, test_loader, args.device),
    )
    log.info(
        "Target Model on Poisoned Data: %s",
        get_accuracy(target_model, poisoned_test_loader, args.device),
    )
    log.info(
        "Poisoned Model on Origin Data: %s",
        get_accuracy(poisoned_model, test_loader, args.device),
    )
    log.info(
        "Poisoned Model on Poisoned Data: %s",
        get_accuracy(poisoned_model, poisoned_test_loader, args.device),
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
