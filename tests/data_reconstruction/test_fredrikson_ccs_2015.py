"""Integration smoke for the FredriksonCCS2015 data-reconstruction attack."""

import pytest
import torch

from amulet.data_reconstruction.attacks.fredrikson_ccs_2015 import FredriksonCCS2015


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

    def run() -> list[torch.Tensor]:
        wrapped_model = SoftmaxModel(tiny_classifier)
        attack = FredriksonCCS2015(
            target_model=wrapped_model,
            input_size=(1, 4),
            output_size=2,
            device=device,
            alpha=5,  # small iterations
            lamda=0.01,
        )
        return attack.attack()

    reconstructed = run()

    assert len(reconstructed) == 2
    assert isinstance(reconstructed[0], torch.Tensor)
    assert reconstructed[0].shape == (1, 4)
    assert reconstructed[1].shape == (1, 4)
    assert torch.isfinite(reconstructed[0]).all()
    assert torch.isfinite(reconstructed[1]).all()

    # Zero-initialized, eval()-mode, no-RNG gradient descent: rerunning the
    # attack on the same model must reproduce the reconstruction bit-exactly.
    reconstructed_again = run()
    assert torch.equal(reconstructed[0], reconstructed_again[0])
    assert torch.equal(reconstructed[1], reconstructed_again[1])
