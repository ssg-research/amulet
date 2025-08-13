import sys

sys.path.append("../../")
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from amulet.poisoning.attacks import BadNets
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
        "--epochs", type=int, default=2, help="Number of epochs for training."
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
    filename = f"{args.dataset}_{args.model}_{args.model_capacity}_{args.training_size*100}_{args.batch_size}_{args.epochs}_{args.exp_id}.pt"

    # Train or Load Target Model
    target_model_path = models_path / "target"
    target_model_filename = target_model_path / filename
    criterion = torch.nn.CrossEntropyLoss()

    if target_model_filename.exists():
        log.info("Target model loaded from %s", target_model_filename)
        target_model = torch.load(target_model_filename).to(args.device)
        optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)
    else:
        log.info("Training target model")
        target_model = initialize_model(
            args.model, args.model_capacity, data.num_features, data.num_classes, log
        ).to(args.device)
        optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)
        target_model = train_classifier(
            target_model,
            target_train_loader,
            criterion,
            optimizer,
            args.epochs,
            args.device,
        )
        log.info("Target model trained")

        # Save model
        create_dir(target_model_path, log)
        torch.save(target_model, target_model_filename)

    test_accuracy_target = get_accuracy(target_model, test_loader, args.device)
    log.info("Test accuracy of target model: %s", test_accuracy_target)

    # Train or Load model for Model Poisoning
    poisoned_model_path = (
        models_path / "model_poisoning" / f"poisoned_protion_{args.poisoned_portion}"
    )
    poisoned_model_filename = poisoned_model_path / filename

    if args.dataset in ["census", "lfw"]:
        dataset_type = "tabular"
    else:
        dataset_type = "image"

    if poisoned_model_filename.exists():
        log.info("Attack model loaded from %s", poisoned_model_filename)
        poisoned_model = torch.load(poisoned_model_filename)
        optimizer = torch.optim.Adam(poisoned_model.parameters(), lr=1e-3)
        poisoning = BadNets(
            args.trigger_label,
            args.poisoned_portion,
            args.exp_id,
            dataset_type,
        )

        poisoned_test_set = poisoning.attack(data.test_set, mode="test")
        poisoned_test_loader = DataLoader(
            dataset=poisoned_test_set, batch_size=args.batch_size, shuffle=False
        )
    else:
        log.info("Running Model Poisoning attack")
        poisoned_model = initialize_model(
            args.model, args.model_capacity, data.num_features, data.num_classes, log
        ).to(args.device)
        optimizer = torch.optim.Adam(poisoned_model.parameters(), lr=1e-3)
        poisoning = BadNets(
            args.trigger_label,
            args.poisoned_portion,
            args.exp_id,
            dataset_type,
        )

        poisoned_train_set = poisoning.attack(data.train_set)
        poisoned_train_loader = DataLoader(
            dataset=poisoned_train_set, batch_size=args.batch_size, shuffle=False
        )

        poisoned_model = train_classifier(
            poisoned_model,
            poisoned_train_loader,
            criterion,
            optimizer,
            args.epochs,
            args.device,
        )

        poisoned_test_set = poisoning.attack(data.test_set, mode="test")
        poisoned_test_loader = DataLoader(
            dataset=poisoned_test_set, batch_size=args.batch_size, shuffle=False
        )

        # Save model
        create_dir(poisoned_model_path, log)
        torch.save(poisoned_model, poisoned_model_filename)

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
