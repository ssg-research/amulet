"""Tests for SuriEvans2022: the _kl_predictions numerical helper (fast tier)
and the attack lifecycle smokes (@integration).

_kl_predictions subsamples adversary-model pairs via the *global* numpy RNG
(np.random.permutation), so those tests seed through seed_everything.
"""

import numpy as np
import pytest

from amulet.distribution_inference.attacks.suri_evans_2022 import SuriEvans2022

N_ADV_MODELS = 4
N_VIC_MODELS = 3
N_SAMPLES = 6
# _kl_predictions keeps 80% of the C(N_ADV_MODELS, 2) upper-triangle pairs.
N_PAIRS = int(0.8 * (N_ADV_MODELS * (N_ADV_MODELS - 1) // 2))


def test_kl_predictions_finite_on_saturated_logits(seed_everything) -> None:
    """Logits large enough that sigmoid rounds to exactly 0/1 must not produce
    NaN/inf — the eps-clip before the Bernoulli KL is what keeps this finite."""
    seed_everything(0)
    rng = np.random.default_rng(0)
    adv_1 = rng.choice([-1e4, 1e4], size=(N_ADV_MODELS, N_SAMPLES))
    adv_2 = rng.choice([-1e4, 1e4], size=(N_ADV_MODELS, N_SAMPLES))
    vic_1 = rng.choice([-1e4, 1e4], size=(N_VIC_MODELS, N_SAMPLES))
    vic_2 = rng.choice([-1e4, 1e4], size=(N_VIC_MODELS, N_SAMPLES))

    preds_first, preds_second = SuriEvans2022._kl_predictions(
        adv_1, adv_2, vic_1, vic_2
    )

    assert preds_first.shape == (N_VIC_MODELS, N_PAIRS)
    assert preds_second.shape == (N_VIC_MODELS, N_PAIRS)
    assert np.isfinite(preds_first).all()
    assert np.isfinite(preds_second).all()


def test_kl_predictions_zero_when_all_models_agree(seed_everything) -> None:
    """When every model in every population emits the same logits, each KL term
    is KL(p, p) = 0 and the distinguishing scores vanish.

    The single logit row is tiled across the model axis: equal *arrays* alone
    are insufficient, because each pairwise score subtracts KLs taken against
    two different adversary models, which only cancel when the rows agree too.
    """
    seed_everything(0)
    rng = np.random.default_rng(0)
    adv_logits = np.tile(rng.standard_normal((1, N_SAMPLES)), (N_ADV_MODELS, 1))
    vic_logits = adv_logits[:N_VIC_MODELS].copy()

    preds_first, preds_second = SuriEvans2022._kl_predictions(
        adv_logits, adv_logits.copy(), vic_logits, vic_logits.copy()
    )

    np.testing.assert_allclose(preds_first, 0.0, atol=1e-12)
    np.testing.assert_allclose(preds_second, 0.0, atol=1e-12)


@pytest.fixture
def suri_evans_factory(tmp_path, synthetic_data_factory):
    """Factory fixture: SuriEvans2022 on the shared synthetic bundle."""

    def _make(device: str = "cpu") -> SuriEvans2022:
        x_train, y_train, z_train, x_test, y_test, z_test = synthetic_data_factory()
        return SuriEvans2022(
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
            num_features=4,
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

    return _make


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_suri_evans_distribution_inference_smoke(suri_evans_factory, device):
    attack = suri_evans_factory(device=device)

    attack.prepare_model_populations()
    results = attack.attack()

    assert "predictions" in results
    assert "ground_truth" in results
    assert len(results["predictions"]) > 0
    assert len(results["predictions"]) == len(results["ground_truth"])


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_suri_evans_reproducible(suri_evans_factory, seed_everything, cpu_device):
    """The only stochastic step in attack() is np.random.permutation over the
    pairwise indices (the global legacy RNG), so with a single fixed population
    seeding numpy before each attack() call reproduces the scores exactly."""
    attack = suri_evans_factory(device=cpu_device)
    attack.prepare_model_populations()

    seed_everything(3)
    first = attack.attack()
    seed_everything(3)
    second = attack.attack()

    np.testing.assert_array_equal(first["predictions"], second["predictions"])
