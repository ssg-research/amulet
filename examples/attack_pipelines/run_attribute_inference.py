import sys

sys.path.append("../../")
import argparse
import logging
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from amulet.attribute_inference.attacks import DudduCIKM2022
from amulet.attribute_inference.metrics import evaluate_attribute_inference
from amulet.utils import (
    load_data,
    initialize_model,
    train_classifier,
    create_dir,
    get_accuracy,
)
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="../../",
        help="Root directory of models and datasets.",
    )
    parser.add_argument(
        "--dataset", type=str, default="census", help="Options: lfw, census."
    )
    parser.add_argument(
        "--model", type=str, default="binarynet", help="Options: binarynet."
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

    if (
        data.x_train is None
        or data.y_train is None
        or data.x_test is None
        or data.y_test is None
    ):
        raise Exception("Missing Numpy Arrays in dataset")

    if data.z_train is None or data.z_test is None:
        raise Exception("Dataset has no sensitive attributes")

    split_data = train_test_split(
        data.x_train, data.y_train, data.z_train, test_size=args.adv_train_fraction
    )

    (
        x_train_target,
        x_train_adv,
        y_train_target,
        _,
        _,
        z_train_adv,
    ) = split_data

    x_train_target = np.array(x_train_target)
    x_train_adv = np.array(x_train_adv)
    y_train_target = np.array(y_train_target)
    z_train_adv = np.array(z_train_adv)

    # Create data loaders
    target_train_set = TensorDataset(
        torch.from_numpy(x_train_target).type(torch.float),
        torch.from_numpy(y_train_target).type(torch.long),
    )

    test_set = TensorDataset(
        torch.from_numpy(data.x_test).type(torch.float),
        torch.from_numpy(data.y_test).type(torch.long),
    )

    target_train_loader = DataLoader(
        dataset=target_train_set, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False
    )

    # Set up filename and directories to save/load models
    models_path = root_dir / "saved_models"
    filename = f"{args.dataset}_{args.model}_{args.model_capacity}_{args.training_size*100}_{args.batch_size}_{args.epochs}_{args.exp_id}.pt"

    # Train or Load Target Model
    target_model_path = (
        models_path
        / "targetLimitedData"
        / f"adv_train_fraction_{args.adv_train_fraction}"
    )
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

    # Run Attribute Inference attack
    attribute_inference = DudduCIKM2022(
        target_model, x_train_adv, data.x_test, z_train_adv, args.device
    )
    predictions = attribute_inference.attack_predictions()

    results = evaluate_attribute_inference(data.z_test, predictions)

    print(results)


if __name__ == "__main__":
    args = parse_args()
    main(args)
