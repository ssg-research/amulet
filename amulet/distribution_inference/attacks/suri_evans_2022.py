"""
KL-divergence-based distribution inference attack.

Reference:
    Anshuman Suri and David Evans.
    "Formalizing and Estimating Distribution Inference Risks."
    NeurIPS 2022. https://arxiv.org/abs/2109.06024
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from torch.distributions.kl import kl_divergence
from torch.utils.data import DataLoader
from tqdm import tqdm

from .distribution_inference_attack import DistributionInferenceAttack


class SuriEvans2022(DistributionInferenceAttack):
    """
    KL-divergence distinguishing test between two training distributions.

    Given two adversary model populations (one per distribution) and two victim
    model populations, the attack measures how each victim's output distribution
    diverges from each adversary baseline. The resulting pairwise KL differences
    are min-max normalized into continuous distinguishing scores in [0, 1], one
    per victim model.

    Inherits all constructor parameters from DistributionInferenceAttack.
    Call prepare_model_populations() before attack(), or pass populations and
    loaders explicitly to attack().
    """

    def _collect_predictions(
        self, loader: DataLoader, models: list[nn.Module]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run each model over loader and stack the class-0 logits."""
        ground_truth_chunks: list[np.ndarray] = []
        for batch in loader:
            ground_truth_chunks.append(batch[1].cpu().numpy())
        ground_truth = np.concatenate(ground_truth_chunks, axis=0)

        per_model_preds: list[np.ndarray] = []
        for model in tqdm(models, desc="Generating Predictions"):
            model = model.to(self.device)
            model.eval()
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

            batch_preds: list[torch.Tensor] = []
            with torch.no_grad():
                for batch in loader:
                    x = batch[0].to(self.device)
                    batch_preds.append(model(x).detach()[:, 0])
            per_model_preds.append(torch.cat(batch_preds).cpu().numpy())

            model.cpu()
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

        return np.stack(per_model_preds, axis=0), ground_truth

    @staticmethod
    def _check_finite(x: np.ndarray) -> None:
        if np.any(np.isinf(x)) or np.any(np.isnan(x)):
            raise ValueError("KL divergence produced non-finite values.")

    @staticmethod
    def _pairwise_compare(
        x: np.ndarray, y: np.ndarray, xx: np.ndarray, yy: np.ndarray
    ) -> np.ndarray:
        x_ = np.expand_dims(x, 2)
        y_ = np.transpose(np.expand_dims(y, 2), (0, 2, 1))
        pairwise = x_ - y_
        return np.array([z[xx, yy] for z in pairwise])

    @staticmethod
    def _kl_predictions(
        adv_preds_1: np.ndarray,
        adv_preds_2: np.ndarray,
        vic_preds_1: np.ndarray,
        vic_preds_2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Turn raw logits into pairwise KL distinguishing scores."""
        adv_1, adv_2, vic_1, vic_2 = (
            torch.sigmoid(torch.from_numpy(p)).numpy()
            for p in (adv_preds_1, adv_preds_2, vic_preds_1, vic_preds_2)
        )

        eps = 1e-4
        log_a = np.log((eps + adv_1) / (eps + 1 - adv_1))
        log_b = np.log((eps + adv_2) / (eps + 1 - adv_2))
        ordering = np.argsort(np.mean(np.abs(log_a - log_b), axis=0))[::-1]
        ordering = ordering[: len(ordering) // 2]
        adv_1, adv_2 = adv_1[:, ordering], adv_2[:, ordering]
        vic_1, vic_2 = vic_1[:, ordering], vic_2[:, ordering]

        # Clip once to keep KL finite; PyTorch's kl_divergence has no eps parameter.
        adv_1_t = torch.from_numpy(np.clip(adv_1, eps, 1 - eps))
        adv_2_t = torch.from_numpy(np.clip(adv_2, eps, 1 - eps))
        vic_1_t = torch.from_numpy(np.clip(vic_1, eps, 1 - eps))
        vic_2_t = torch.from_numpy(np.clip(vic_2, eps, 1 - eps))

        xx, yy = np.triu_indices(adv_preds_1.shape[0], k=1)
        random_pick = np.random.permutation(xx.shape[0])[: int(0.8 * xx.shape[0])]
        xx, yy = xx[random_pick], yy[random_pick]

        # PyTorch's Bernoulli KL has a broadcasting bug — its internal mask
        # indexing fails when the two probs tensors don't share a shape. Expand
        # each victim row to adv's shape before constructing the distributions.
        kl_1_a = np.array([
            kl_divergence(
                Bernoulli(probs=adv_1_t),
                Bernoulli(probs=v.expand_as(adv_1_t)),
            )
            .mean(dim=1)
            .numpy()
            for v in vic_1_t
        ])
        SuriEvans2022._check_finite(kl_1_a)
        kl_1_b = np.array([
            kl_divergence(
                Bernoulli(probs=adv_2_t),
                Bernoulli(probs=v.expand_as(adv_2_t)),
            )
            .mean(dim=1)
            .numpy()
            for v in vic_1_t
        ])
        SuriEvans2022._check_finite(kl_1_b)
        kl_2_a = np.array([
            kl_divergence(
                Bernoulli(probs=adv_1_t),
                Bernoulli(probs=v.expand_as(adv_1_t)),
            )
            .mean(dim=1)
            .numpy()
            for v in vic_2_t
        ])
        SuriEvans2022._check_finite(kl_2_a)
        kl_2_b = np.array([
            kl_divergence(
                Bernoulli(probs=adv_2_t),
                Bernoulli(probs=v.expand_as(adv_2_t)),
            )
            .mean(dim=1)
            .numpy()
            for v in vic_2_t
        ])
        SuriEvans2022._check_finite(kl_2_b)

        preds_first = SuriEvans2022._pairwise_compare(kl_1_a, kl_1_b, xx, yy)
        preds_second = SuriEvans2022._pairwise_compare(kl_2_a, kl_2_b, xx, yy)
        return preds_first, preds_second

    def attack(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        models_adv_1: list[nn.Module] | None = None,
        models_adv_2: list[nn.Module] | None = None,
        models_vic_1: list[nn.Module] | None = None,
        models_vic_2: list[nn.Module] | None = None,
        test_loader_1: DataLoader | None = None,
        test_loader_2: DataLoader | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Run the distinguishing test.

        Args default to the populations and test loaders produced by
        prepare_model_populations(). Pass any subset explicitly to run the
        attack against externally produced populations or loaders.

        Args:
            models_adv_1: Adversary models trained on distribution 1. Defaults to
                the prepared adversary-D1 population.
            models_adv_2: Adversary models trained on distribution 2. Defaults to
                the prepared adversary-D2 population.
            models_vic_1: Victim models trained on distribution 1. Defaults to the
                prepared victim-D1 population.
            models_vic_2: Victim models trained on distribution 2. Defaults to the
                prepared victim-D2 population.
            test_loader_1: Test loader for distribution 1. Defaults to the prepared
                distribution-1 test loader.
            test_loader_2: Test loader for distribution 2. Defaults to the prepared
                distribution-2 test loader.

        Returns:
            A dictionary with two 1-D arrays of equal length:
                - "predictions": attack scores in [0, 1]. Thresholding at 0.5
                  yields the distinguishing decision.
                - "ground_truth": 0 for distribution-1 victims, 1 for
                  distribution-2 victims.

        Raises:
            RuntimeError: If the test loaders are not supplied and
                prepare_model_populations() has not been called, or if any of the
                four model populations is empty.
        """
        models_adv_1 = models_adv_1 if models_adv_1 is not None else self.models_adv_1
        models_adv_2 = models_adv_2 if models_adv_2 is not None else self.models_adv_2
        models_vic_1 = models_vic_1 if models_vic_1 is not None else self.models_vic_1
        models_vic_2 = models_vic_2 if models_vic_2 is not None else self.models_vic_2

        if test_loader_1 is None or test_loader_2 is None:
            if self.splits is None:
                raise RuntimeError(
                    "Pass test_loader_1/test_loader_2 explicitly or call "
                    "prepare_model_populations() first."
                )
            test_loader_1 = test_loader_1 or self.splits.test_loader_1
            test_loader_2 = test_loader_2 or self.splits.test_loader_2

        if not (models_adv_1 and models_adv_2 and models_vic_1 and models_vic_2):
            raise RuntimeError("All four model populations must be non-empty.")

        adv1_on_1, _ = self._collect_predictions(test_loader_1, models_adv_1)
        adv2_on_1, _ = self._collect_predictions(test_loader_1, models_adv_2)
        vic1_on_1, _ = self._collect_predictions(test_loader_1, models_vic_1)
        vic2_on_1, _ = self._collect_predictions(test_loader_1, models_vic_2)

        adv1_on_2, _ = self._collect_predictions(test_loader_2, models_adv_1)
        adv2_on_2, _ = self._collect_predictions(test_loader_2, models_adv_2)
        vic1_on_2, _ = self._collect_predictions(test_loader_2, models_vic_1)
        vic2_on_2, _ = self._collect_predictions(test_loader_2, models_vic_2)

        preds_1_first, preds_1_second = self._kl_predictions(
            adv1_on_1, adv2_on_1, vic1_on_1, vic2_on_1
        )
        preds_2_first, preds_2_second = self._kl_predictions(
            adv1_on_2, adv2_on_2, vic1_on_2, vic2_on_2
        )

        preds_first = np.concatenate((preds_1_first, preds_2_first), axis=1)
        preds_second = np.concatenate((preds_1_second, preds_2_second), axis=1)
        stacked = np.concatenate((preds_first, preds_second))
        stacked -= np.min(stacked, axis=0)
        stacked /= np.max(stacked, axis=0)
        scores = np.mean(stacked, axis=1)

        ground_truth = np.concatenate((
            np.zeros(preds_first.shape[0]),
            np.ones(preds_second.shape[0]),
        ))
        return {"predictions": scores, "ground_truth": ground_truth}
