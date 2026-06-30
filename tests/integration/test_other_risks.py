import numpy as np
import pytest
import torch

from amulet.attribute_inference.attacks.duddu_cikm_2022 import DudduCIKM2022
from amulet.data_reconstruction.attacks.fredrikson_ccs_2015 import FredriksonCCS2015
from amulet.distribution_inference.attacks.suri_evans_2022 import SuriEvans2022


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_duddu_attribute_inference_smoke(tiny_classifier, device):
    x_train_adv = np.random.rand(20, 4)
    x_test = np.random.rand(10, 4)
    z_train_adv = np.random.randint(0, 2, (20, 1))

    attack = DudduCIKM2022(
        target_model=tiny_classifier,
        x_train_adv=x_train_adv,
        x_test=x_test,
        z_train_adv=z_train_adv,
        batch_size=8,
        device=device,
    )

    results = attack.attack()

    assert 0 in results
    assert "predictions" in results[0]
    assert len(results[0]["predictions"]) == 10


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_suri_evans_distribution_inference_smoke(tmp_path, device):
    # Use synthetic tabular data large enough to satisfy the ratio
    # subsampling (600 train rows, 200 test rows, 2 sensitive columns).
    rng = np.random.default_rng(0)
    n_train, n_test, num_features = 600, 200, 4

    x_train = rng.standard_normal((n_train, num_features)).astype(np.float32)
    y_train = rng.integers(0, 2, size=n_train)
    z_train = rng.integers(0, 2, size=(n_train, 2))

    x_test = rng.standard_normal((n_test, num_features)).astype(np.float32)
    y_test = rng.integers(0, 2, size=n_test)
    z_test = rng.integers(0, 2, size=(n_test, 2))

    attack = SuriEvans2022(
        x_train=x_train,
        y_train=y_train,
        z_train=z_train,
        x_test=x_test,
        y_test=y_test,
        z_test=z_test,
        sensitive_columns=["race", "sex"],
        filter_column="sex",
        ratio1=0.1,
        ratio2=0.9,
        model_arch="linearnet",
        model_capacity="m1",
        num_features=num_features,
        num_classes=2,
        num_models=3,
        epochs=1,
        batch_size=16,
        device=device,
        models_dir=tmp_path,
        dataset="synthetic",
        train_subsample=50,
        test_subsample=25,
    )

    attack.prepare_model_populations()

    results = attack.attack()

    assert "predictions" in results
    assert "ground_truth" in results
    assert len(results["predictions"]) > 0
    assert len(results["predictions"]) == len(results["ground_truth"])


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_fredrikson_reconstruction_smoke(tiny_classifier, device):
    # Wrap classifier in softmax as requested by docstring
    class SoftmaxModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return torch.softmax(self.model(x), dim=1)

    wrapped_model = SoftmaxModel(tiny_classifier)

    attack = FredriksonCCS2015(
        target_model=wrapped_model,
        input_size=(1, 4),
        output_size=2,
        device=device,
        alpha=5,  # small iterations
        lamda=0.01,
    )

    reconstructed = attack.attack()

    assert len(reconstructed) == 2
    assert isinstance(reconstructed[0], torch.Tensor)
    assert reconstructed[0].shape == (1, 4)
