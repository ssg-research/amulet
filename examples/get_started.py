"""
Simple end-to-end guide to evaluate your model with Amulet, with and without a defense.

Trains a classifier on CIFAR-10, runs a PGD evasion attack, then defends with
adversarial training and reruns the attack to show the accuracy trade-off.

Run from the repo root:
    uv run python examples/get_started.py
"""

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from amulet.evasion.attacks import EvasionPGD
from amulet.evasion.defenses import AdversarialTrainingPGD
from amulet.utils import (
    get_accuracy,
    initialize_model,
    load_data,
    train_classifier,
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger(__name__)

    root_dir = Path(__file__).parent.parent
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    random_seed = 123
    epochs = 10
    batch_size = 256
    epsilon = 0.1

    torch.manual_seed(random_seed)

    # cifar10 and fmnist download automatically; no external data needed.
    dataset = "cifar10"  # Options: cifar10, fmnist, lfw, census, celeba
    data = load_data(root_dir, dataset)
    train_loader = DataLoader(
        dataset=data.train_set, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        dataset=data.test_set, batch_size=batch_size, shuffle=False
    )

    criterion = torch.nn.CrossEntropyLoss()
    target_model = initialize_model(
        model_arch="vgg",
        model_capacity="m1",
        num_features=data.num_features,
        num_classes=data.num_classes,
    ).to(device)
    optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)
    target_model = train_classifier(
        target_model, train_loader, criterion, optimizer, epochs, device
    )

    test_acc = get_accuracy(target_model, test_loader, device)
    log.info("Clean test accuracy: %.4f", test_acc)

    # Attack undefended model
    evasion = EvasionPGD(
        target_model, test_loader, device, batch_size=batch_size, epsilon=epsilon
    )
    adv_loader = evasion.attack()
    adv_acc = get_accuracy(target_model, adv_loader, device)
    log.info("Adversarial accuracy (undefended): %.4f", adv_acc)

    # Defend with adversarial training, then rerun attack
    adv_training = AdversarialTrainingPGD(
        target_model,
        criterion,
        optimizer,
        train_loader,
        device,
        epochs,
        epsilon=epsilon,
    )
    defended_model = adv_training.train_robust()

    test_acc_defended = get_accuracy(defended_model, test_loader, device)
    log.info("Clean test accuracy (defended): %.4f", test_acc_defended)

    evasion_def = EvasionPGD(
        defended_model, test_loader, device, batch_size=batch_size, epsilon=epsilon
    )
    adv_loader_def = evasion_def.attack()
    adv_acc_defended = get_accuracy(defended_model, adv_loader_def, device)
    log.info("Adversarial accuracy (defended): %.4f", adv_acc_defended)

    log.info(
        "Summary: clean %.4f -> %.4f | adversarial %.4f -> %.4f",
        test_acc,
        test_acc_defended,
        adv_acc,
        adv_acc_defended,
    )


if __name__ == "__main__":
    main()
