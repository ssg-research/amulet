"""Integration smoke for the DudduCIKM2022 attribute-inference attack."""

import numpy as np
import pytest

from amulet.attribute_inference.attacks.duddu_cikm_2022 import DudduCIKM2022


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
    assert "confidence_values" in results[0]
    confidence_values = results[0]["confidence_values"]
    assert len(confidence_values) == 10
    assert (confidence_values >= 0.0).all()
    assert (confidence_values <= 1.0).all()
