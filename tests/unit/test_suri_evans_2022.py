"""Unit tests for SuriEvans2022._kl_predictions.

The helper turns raw per-model logits into pairwise KL distinguishing scores.
It subsamples adversary-model pairs via the *global* numpy RNG
(np.random.permutation), so each test seeds through seed_everything.
"""

import numpy as np

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
