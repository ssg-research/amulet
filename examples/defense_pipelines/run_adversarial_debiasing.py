import sys

sys.path.append("../../")
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from amulet.discriminatory_behavior.defenses import AdversarialDebiasing
from amulet.discriminatory_behavior.metrics import DiscriminatoryBehavior
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
        "--dataset", type=str, default="celeba", help="Options: lfw, census, celeba."
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
        default=torch.device(
            "cuda:{0}".format(0) if torch.cuda.is_available() else "cpu"
        ),
        help="Device on which to run PyTorch",
    )
    parser.add_argument(
        "--exp_id", type=int, default=0, help="Used as a random seed for experiments."
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Setup logger
    root_dir = Path(args.root)
    log_dir = root_dir / "logs"
    create_dir(log_dir)
    logging.basicConfig(
        level=logging.INFO, filename=log_dir / "group_fairness.log", filemode="w"
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

    if data.z_train is None or data.z_test is None:
        raise RuntimeError(
            "Dataset does not contain sensitive attributes. Please check if you are using a supported dataset"
        )

    sensitive_train_set = TensorDataset(
        torch.from_numpy(data.x_train).type(torch.float),
        torch.from_numpy(data.y_train).type(torch.long),
        torch.from_numpy(data.z_train).type(torch.float),
    )

    sensitive_test_set = TensorDataset(
        torch.from_numpy(data.x_test).type(torch.float),
        torch.from_numpy(data.y_test).type(torch.long),
        torch.from_numpy(data.z_test).type(torch.float),
    )

    sensitive_train_loader = DataLoader(
        dataset=sensitive_train_set, batch_size=args.batch_size, shuffle=False
    )
    sensitive_test_loader = DataLoader(
        dataset=sensitive_test_set, batch_size=args.batch_size, shuffle=False
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
            args.model, args.model_capacity, data.num_features, data.num_classes, log
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

    # Measure discriminatory behavior of target model
    discr_behavior_target = DiscriminatoryBehavior(
        target_model, sensitive_test_loader, args.device
    )
    all_metrics = discr_behavior_target.evaluate_subgroup_metrics()

    for attribute, metrics in all_metrics.items():
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

    # Train or load model with Adversarial Training
    defended_model_path = models_path / "adversarial_debiasing"
    defended_model_filename = defended_model_path / filename

    if defended_model_filename.exists():
        log.info("Defended model loaded from %s", defended_model_filename)
        defended_model = torch.load(defended_model_filename)
    else:
        log.info("Retraining Model with Group Fairness")
        if (
            args.dataset == "lfw"
        ):  # change lambdas manually to get better trade-off; hyperparameter tuning is hard
            lambdas = torch.Tensor([45, 17])
        else:
            lambdas = torch.Tensor([40, 40])

        group_fairness = AdversarialDebiasing(
            target_model,
            criterion,
            optimizer,
            sensitive_train_loader,
            sensitive_test_loader,
            data.z_train.shape[1],
            data.num_classes,
            lambdas,
            args.device,
            args.epochs,
        )
        defended_model = group_fairness.train_fair()
        # Save model
        create_dir(defended_model_path, log)
        torch.save(defended_model, defended_model_filename)

    # Measure discriminatory behavior of defended model
    discr_behavior_defended = DiscriminatoryBehavior(
        defended_model, sensitive_test_loader, args.device
    )
    all_metrics = discr_behavior_defended.evaluate_subgroup_metrics()

    for attribute, metrics in all_metrics.items():
        print(attribute)
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
    test_accuracy_defended = get_accuracy(defended_model, test_loader, args.device)
    log.info("Test accuracy of defended model: %s", test_accuracy_defended)


if __name__ == "__main__":
    args = parse_args()
    main(args)
