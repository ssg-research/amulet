"""Utilities to extract model parameters as features for meta-inference."""

from typing import cast

import torch
import torch.nn as nn


def get_layer_parameters(model: nn.Module) -> list[torch.Tensor]:
    """
    Extract weights and biases from Linear and Conv2d layers.

    Captures the parameters of the model and formats them as a list of tensors,
    where each tensor represents one layer. The output is formatted to support
    permutation-invariant processing (one row per neuron/channel).

    Args:
        model: Model to extract parameters from.

    Returns:
        List of tensors, one per layer, with shape [N_neurons, Dim_per_neuron].
    """
    features = []
    # Unwrap DataParallel/DistributedDataParallel if present
    base_model = cast(nn.Module, model.module if hasattr(model, "module") else model)

    for _, module in base_model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Weight shape for Linear: [out_features, in_features]
            # Weight shape for Conv2d: [out_channels, in_channels, k_h, k_w]
            w = module.weight.data.detach().cpu()

            if isinstance(module, nn.Conv2d):
                # Flatten Conv kernels: [out, in*k*k]
                w = w.view(w.size(0), -1)

            # Append bias if it exists: [out, feat] -> [out, feat + 1]
            if module.bias is not None:
                b = module.bias.data.detach().cpu().unsqueeze(1)
                layer_feat = torch.cat([w, b], dim=1)
            else:
                layer_feat = w

            features.append(layer_feat)

    return features
