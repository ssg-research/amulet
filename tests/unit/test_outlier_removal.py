"""Unit test for OutlierRemoval._knn_shapley, the numerical core of the defense.

The recursion that fills the Shapley matrix is the load-bearing math; the
surrounding train_robust pipeline (feature extraction + retraining) is covered
by the integration smoke in tests/integration/test_poisoning.py.
"""

import numpy as np
import torch
import torch.nn as nn

from amulet.poisoning.defenses.outlier_removal import OutlierRemoval

N_TRAIN = 8
N_TEST = 5
DIM = 3


def test_knn_shapley_returns_finite_score_per_train_point(
    tiny_classifier_factory, tiny_loader, cpu_device
) -> None:
    model = tiny_classifier_factory(seed=0)
    defense = OutlierRemoval(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        train_loader=tiny_loader,
        test_loader=tiny_loader,
        device=cpu_device,
    )
    rng = np.random.default_rng(0)

    scores = defense._knn_shapley(
        train_features=rng.standard_normal((N_TRAIN, DIM)),
        train_targets=rng.integers(0, 2, size=N_TRAIN),
        test_features=rng.standard_normal((N_TEST, DIM)),
        test_targets=rng.integers(0, 2, size=N_TEST),
    )

    assert scores.shape == (N_TRAIN,)
    assert np.isfinite(scores).all()
