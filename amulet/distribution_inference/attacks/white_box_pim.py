"""White-Box Distribution Inference using Permutation Invariant Models (PIMs)."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ...utils.__meta_classifiers import PermInvModel, meta_collate_fn
from .distribution_inference_attack import DistributionInferenceAttack


def _get_layer_parameters(model: nn.Module) -> list[torch.Tensor]:
    """
    Format a model's Linear and Conv2d parameters as PIM meta-classifier input.

    Each extracted layer becomes one 2-D tensor of shape
    [N_neurons, Dim_per_neuron] — the permutation-invariant row-per-neuron
    layout PermInvModel expects. Conv2d kernels are flattened to
    [out, in*k*k]; a bias, when present, is appended as one extra column.
    All other layer types are skipped.

    Args:
        model: Model to extract parameters from.

    Returns:
        List of 2-D tensors, one per Linear/Conv2d layer.

    Raises:
        ValueError: If the model contains no Linear or Conv2d layers, which
            would otherwise fail far from the cause when the meta-classifier
            is built over an empty shape list.
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

    if not features:
        raise ValueError(
            "Model has no Linear or Conv2d layers to extract parameters from; "
            "the PIM meta-classifier needs at least one."
        )

    return features


class WhiteBoxPIM(DistributionInferenceAttack):
    """
    White-box distribution inference attack using Permutation Invariant Models.

    Extracts the raw weights of adversary-population models, trains a
    Permutation Invariant Model (PIM) meta-classifier to distinguish
    distribution 1 from distribution 2, then evaluates it on held-out
    target models.

    Adversary populations supply the meta-classifier's training data.
    Target populations are the evaluation set that determines the attack's
    distinguishing accuracy — they must not overlap with the adversary sets.

    Reference:
        Anshuman Suri and David Evans.
        "Formalizing and Estimating Distribution Inference Risks."
        NeurIPS 2022. https://arxiv.org/abs/2109.06024

    Attributes:
        meta_epochs: Number of training epochs for the PIM meta-classifier.
        lr: Learning rate for the meta-classifier optimiser.
        inside_dims: Hidden dimensions for the PIM's per-layer sub-networks.
            Passed directly to PermInvModel; defaults to [64, 8] when None.
        random_seed: Optional seed applied before meta-classifier training
            for reproducibility. Does not affect population-model training.
    """

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        z_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        z_test: np.ndarray,
        sensitive_columns: list[str],
        filter_column: str,
        ratio1: float,
        ratio2: float,
        model_arch: str,
        model_capacity: str,
        num_features: int,
        num_classes: int,
        num_models: int,
        epochs: int,
        batch_size: int,
        device: str,
        models_dir: Path | str,
        dataset: str,
        exp_id: int = 0,
        filter_value: int = 1,
        drop_values: dict[str, list[int]] | None = None,
        train_subsample: int = 1100,
        test_subsample: int = 500,
        meta_epochs: int = 100,
        lr: float = 1e-2,
        inside_dims: list[int] | None = None,
        random_seed: int | None = None,
    ):
        super().__init__(
            x_train,
            y_train,
            z_train,
            x_test,
            y_test,
            z_test,
            sensitive_columns,
            filter_column,
            ratio1,
            ratio2,
            model_arch,
            model_capacity,
            num_features,
            num_classes,
            num_models,
            epochs,
            batch_size,
            device,
            models_dir,
            dataset,
            exp_id,
            filter_value,
            drop_values,
            train_subsample,
            test_subsample,
        )
        self.meta_epochs = meta_epochs
        self.lr = lr
        self.inside_dims = inside_dims
        self.random_seed = random_seed
        self._metamodel: PermInvModel | None = None

    def _build_dataset(
        self,
        models_1: list[nn.Module],
        models_2: list[nn.Module],
    ) -> list[tuple[list[torch.Tensor], int]]:
        dataset: list[tuple[list[torch.Tensor], int]] = []
        for m in models_1:
            dataset.append((_get_layer_parameters(m), 0))
        for m in models_2:
            dataset.append((_get_layer_parameters(m), 1))
        return dataset

    def _make_loader(
        self,
        dataset: list[tuple[list[torch.Tensor], int]],
        shuffle: bool,
    ) -> DataLoader[Any]:
        return DataLoader(
            cast(Dataset[Any], dataset),
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=meta_collate_fn,
        )

    def attack(self) -> dict[str, np.ndarray]:
        """
        Train the PIM meta-classifier on adversary models and evaluate on targets.

        Must call prepare_model_populations() first.

        Returns:
            Dictionary with two 1-D arrays of equal length:
                - "predictions": sigmoid scores in [0, 1]. Values close to 1
                  indicate the model is more likely from distribution 2.
                - "ground_truth": 0 for distribution-1 targets, 1 for
                  distribution-2 targets.

        Raises:
            RuntimeError: If prepare_model_populations() has not been called.
        """
        if not self.models_adv_1:
            raise RuntimeError("Call prepare_model_populations() before attack().")

        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)

        train_data = self._build_dataset(self.models_adv_1, self.models_adv_2)
        eval_data = self._build_dataset(self.models_target_1, self.models_target_2)

        train_loader = self._make_loader(train_data, shuffle=True)
        eval_loader = self._make_loader(eval_data, shuffle=False)

        # _get_layer_parameters guarantees non-empty, 2-D tensors.
        layer_shapes = [(p.shape[0], p.shape[1]) for p in train_data[0][0]]

        metamodel = PermInvModel(
            layer_shapes=layer_shapes,
            inside_dims=self.inside_dims,
        ).to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(metamodel.parameters(), lr=self.lr, weight_decay=1e-2)

        metamodel.train()
        for _ in range(self.meta_epochs):
            for layer_params, labels in train_loader:
                layer_params = [p.to(self.device) for p in layer_params]
                labels = labels.to(self.device).float().unsqueeze(1)
                optimizer.zero_grad()
                loss = criterion(metamodel(layer_params), labels)
                loss.backward()
                optimizer.step()

        self._metamodel = metamodel

        metamodel.eval()
        predictions: list[float] = []
        with torch.no_grad():
            for layer_params, _ in eval_loader:
                layer_params = [p.to(self.device) for p in layer_params]
                scores = torch.sigmoid(metamodel(layer_params)).squeeze(1)
                predictions.extend(scores.cpu().tolist())

        ground_truth = np.concatenate([
            np.zeros(len(self.models_target_1)),
            np.ones(len(self.models_target_2)),
        ])
        return {
            "predictions": np.array(predictions),
            "ground_truth": ground_truth,
        }
