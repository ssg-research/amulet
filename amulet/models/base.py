"""Shared base class for Amulet models."""

import torch
import torch.nn as nn


class AmuletModel(nn.Module):
    """Base class for models in `amulet/models/`.

    Amulet pipelines (e.g. `get_intermediate_features`, `OutlierRemoval`) rely on
    models exposing a `get_hidden` method that returns intermediate-layer activations.
    Subclasses must override both `forward` and `get_hidden`; the default
    `get_hidden` here mirrors `nn.Module.forward` and raises `NotImplementedError`
    so the contract is enforced at runtime without an abstract-method metaclass.
    """

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} must implement `get_hidden`.")
