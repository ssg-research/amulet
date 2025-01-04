import sys

sys.path.append("../../")
import argparse
import logging
from pathlib import Path

import torch
from amulet.distribution_inference.attacks import DistributionInference
from amulet.utils import load_data, train_classifier, create_dir, get_accuracy
from amulet.models.linear_net import LinearNet


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
        "--model", type=str, default="linearnet", help="Options: linearnet."
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
    parser.add_argument("--ratio1", help="Ratio for D_0", default="0.1")
    parser.add_argument("--ratio2", help="Ratio for D_0", default="0.9")
    parser.add_argument(
        "--filter", type=str, default="race", help="while filter to use"
    )
    parser.add_argument(
        "--num_models",
        type=int,
        default=5,
        help="Used as a random seed for experiments.",
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Setup logger
    root_dir = Path(args.root)
    log_dir = root_dir / "logs"
    create_dir(log_dir)
    logging.basicConfig(
        level=logging.INFO,
        filename=log_dir / "distribution_inference.log",
        filemode="w",
    )
    log = logging.getLogger("All")
    log.addHandler(logging.StreamHandler())

    # Set random seeds for reproducibility
    torch.manual_seed(args.exp_id)

    # Load dataset and create data loaders
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

    distinf = DistributionInference(
        data.x_train,
        data.x_test,
        data.y_train,
        data.y_test,
        data.z_train,
        data.z_test,
        args.filter,
        args.ratio1,
        args.ratio2,
        args.device,
        args,
    )
    (
        vic_trainloader_1,
        vic_trainloader_2,
        att_trainloader_1,
        att_trainloader_2,
        test_loader_1,
        test_loader_2,
    ) = distinf.prepare_dataset()

    def train_models(train_loader, test_loader, args):
        trained_models_list = []
        for _ in range(args.num_models):
            target_model = LinearNet(
                num_features=data.num_features,
                num_classes=data.num_classes,
                hidden_layer_sizes=[32, 64, 32],
            ).to(args.device)
            optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)
            criterion = torch.nn.CrossEntropyLoss()
            target_model = train_classifier(
                target_model,
                train_loader,
                criterion,
                optimizer,
                args.epochs,
                args.device,
            )
            test_accuracy_target = get_accuracy(target_model, test_loader, args.device)
            log.info("Test accuracy of target model: %s", test_accuracy_target)
            trained_models_list.append(target_model)
        return trained_models_list

    models_adv_1 = train_models(att_trainloader_1, test_loader_1, args)
    models_adv_2 = train_models(att_trainloader_2, test_loader_1, args)
    models_vic_1 = train_models(vic_trainloader_1, test_loader_1, args)
    models_vic_2 = train_models(vic_trainloader_2, test_loader_1, args)

    # train models
    accuracy = distinf.attack(
        models_vic_1,
        models_vic_2,
        models_adv_1,
        models_adv_2,
        test_loader_1,
        test_loader_2,
    )
    print("Accuracy: {:.2f}".format(accuracy))


if __name__ == "__main__":
    args = parse_args()
    main(args)
