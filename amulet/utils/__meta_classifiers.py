"""Neural architectures for meta-inference (Inference about models)."""

import torch
import torch.nn as nn


class PermInvModel(nn.Module):
    """
    Permutation Invariant Model (PIM) for processing model parameters.

    Processes layers of a model (weights and biases) while maintaining
    invariance to the permutation of neurons within those layers. Each
    neuron is processed through a mini-network, and the results are
    aggregated using sum-pooling.

    Attributes:
        layer_shapes: List of (N_neurons, Dim_per_neuron) for each layer.
        inside_dims: Hidden dimensions for the mini-networks.
        dropout: Dropout rate for regularization.
        n_classes: Number of output classes for the meta-classifier.
    """

    def __init__(
        self,
        layer_shapes: list[tuple[int, int]],
        inside_dims: list[int] | None = None,
        dropout: float = 0.5,
        n_classes: int = 1,
    ):
        super().__init__()
        self.layer_shapes = layer_shapes
        self.inside_dims = inside_dims or [64, 8]
        self.dropout = dropout

        self.mini_nets = nn.ModuleList()
        prev_rep_size = 0

        for _n_neurons, dim in layer_shapes:
            # Each neuron sees its own weights (dim) + sum-pooled representation
            # from the previous layer (prev_rep_size).
            input_dim = dim + prev_rep_size
            self.mini_nets.append(self._make_mini_net(input_dim))
            # The next layer will see H as context (pooled)
            prev_rep_size = self.inside_dims[-1]

        # Final classifier to combine aggregated representations from all layers
        self.rho = nn.Linear(len(layer_shapes) * self.inside_dims[-1], n_classes)

    def _make_mini_net(self, input_dim: int) -> nn.Sequential:
        layers = [nn.Linear(input_dim, self.inside_dims[0]), nn.ReLU()]
        for i in range(1, len(self.inside_dims)):
            layers.append(nn.Linear(self.inside_dims[i - 1], self.inside_dims[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout))
        return nn.Sequential(*layers)

    def forward(self, layer_params_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the PIM.

        Args:
            layer_params_list: List of tensors, one per layer,
                               each with shape [Batch, N_neurons, Dim].

        Returns:
            Logits for the meta-classification decision.

        Raises:
            ValueError: If a layer's neuron count does not match its expected shape.
        """
        reps = []
        for_prev = None

        for _i, (param, net) in enumerate(
            zip(layer_params_list, self.mini_nets, strict=True)
        ):
            # param shape: [Batch, N_neurons, Dim]
            if param.size(1) != self.layer_shapes[_i][0]:
                raise ValueError(
                    f"Layer {_i} expects {self.layer_shapes[_i][0]} neurons, "
                    f"got {param.size(1)}"
                )

            if for_prev is not None:
                # for_prev is [Batch, 1, H] (sum-pooled context from previous layer)
                # Repeat for each neuron in the current layer
                prev_rep = for_prev.repeat(1, param.size(1), 1)
                param_eff = torch.cat([param, prev_rep], dim=-1)
            else:
                param_eff = param

            # Flatten batch and neuron dims for the mini-net
            batch_size, n_neurons, dim_eff = param_eff.shape
            pp = net(param_eff.view(batch_size * n_neurons, dim_eff))
            pp = pp.view(batch_size, n_neurons, -1)  # [Batch, N_neurons, H]

            # Sum-pool across the neuron dimension (Invariance step!)
            processed = torch.sum(pp, dim=1)  # [Batch, H]
            reps.append(processed)

            # Pass aggregated representation as context for the next layer
            # This maintains invariance because 'processed' is already pooled.
            for_prev = processed.view(batch_size, 1, -1)  # [Batch, 1, H]

        combined = torch.cat(reps, dim=1)  # [Batch, Layers * H]
        return self.rho(combined)


def meta_collate_fn(batch: list[tuple[list[torch.Tensor], int]]):
    """
    Custom collate function for model-as-data batches.

    Args:
        batch: List of (layer_params, label) tuples.
               layer_params is a list of [N_neurons, Dim] tensors.

    Returns:
        Tuple of (batched_layer_params, labels).
        batched_layer_params is a list of [Batch, N_neurons, Dim] tensors.
    """
    labels = torch.tensor([item[1] for item in batch])
    num_layers = len(batch[0][0])
    batched_params = []

    for i in range(num_layers):
        layer_batch = torch.stack([item[0][i] for item in batch])
        batched_params.append(layer_batch)

    return batched_params, labels
