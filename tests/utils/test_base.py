"""Tests for the core training primitives in amulet/utils/__base.py:

- train_classifier: gradient flow works end-to-end — loss on a single learnable
  batch collapses (bucket 2: a property of the optimizer, not the data).
- load_or_train: cache-miss trains and writes the checkpoint; cache-hit loads
  it and never calls the training closure.
- get_predictions_numpy: output shape, and invariance to how the input is
  batched.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from amulet.utils import get_predictions_numpy, load_or_train, train_classifier


def test_train_classifier_overfits_single_batch(
    tiny_classifier_factory, cpu_device
) -> None:
    model = tiny_classifier_factory(seed=42)
    # Two well-separated Gaussian blobs rather than random labels: collapse on a
    # learnable batch is a property of the optimizer, while memorising random
    # labels in 50 steps is a seed lottery at this model width.
    x = torch.cat([torch.randn(4, 4) + 1.5, torch.randn(4, 4) - 1.5])
    y = torch.cat([torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.long)])
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    with torch.no_grad():
        initial = criterion(model(x), y).item()

    # patience == epochs: early stopping keys off accuracy and must not be able
    # to truncate the run before the loss has collapsed.
    train_classifier(
        model,
        loader,
        criterion,
        optimizer,
        epochs=50,
        device=cpu_device,
        early_stopping_patience=50,
    )

    with torch.no_grad():
        final = criterion(model(x), y).item()

    assert final < initial * 0.1, f"loss did not collapse: {initial:.4f} -> {final:.4f}"


def _fill_weights(model: nn.Module, value: float) -> nn.Module:
    """Stand-in for a training closure: overwrite every parameter with a known
    constant so 'the checkpoint holds the trained state' is exactly checkable."""
    with torch.no_grad():
        for p in model.parameters():
            _ = p.fill_(value)
    return model


def test_load_or_train_cache_miss_trains_and_saves_checkpoint(
    tmp_path, tiny_classifier_factory, mocker
) -> None:
    path = tmp_path / "checkpoints" / "model.pt"
    init_fn = mocker.Mock(side_effect=lambda: tiny_classifier_factory())
    train_fn = mocker.Mock(side_effect=lambda m: _fill_weights(m, 0.5))

    load_or_train(path, init_fn, train_fn)

    init_fn.assert_called_once()
    train_fn.assert_called_once()
    saved = torch.load(path, weights_only=True)
    assert all((tensor == 0.5).all() for tensor in saved.values())


def test_load_or_train_cache_hit_loads_checkpoint_without_training(
    tmp_path, tiny_classifier_factory, mocker
) -> None:
    path = tmp_path / "model.pt"
    torch.save(_fill_weights(tiny_classifier_factory(), 0.5).state_dict(), path)
    init_fn = mocker.Mock(side_effect=lambda: tiny_classifier_factory())
    train_fn = mocker.Mock()

    model = load_or_train(path, init_fn, train_fn)

    init_fn.assert_called_once()
    train_fn.assert_not_called()
    # A freshly initialised model would have random weights; all-0.5 parameters
    # prove the state came from the checkpoint on disk.
    assert all((p == 0.5).all() for p in model.parameters())


def test_get_predictions_numpy_shape_and_batch_size_invariance(
    tiny_classifier_factory, cpu_device
) -> None:
    model = tiny_classifier_factory(seed=42)
    rng = np.random.default_rng(0)
    input_data = rng.standard_normal((10, 4), dtype=np.float32)

    single_batch = get_predictions_numpy(
        input_data, model, batch_size=10, device=cpu_device
    )
    split_batches = get_predictions_numpy(
        input_data, model, batch_size=3, device=cpu_device
    )

    assert single_batch.shape == (10, 2)
    # Exact equality is unattainable: CPU BLAS picks a different accumulation
    # strategy per batch size, giving sub-ulp float32 differences (~3e-8
    # observed). atol=1e-6 absorbs that while still failing on any real
    # batching bug, which would shift logits by orders of magnitude more.
    np.testing.assert_allclose(split_batches, single_batch, atol=1e-6)
