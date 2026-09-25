"""Implementation of Likelihood Ratio Attack (LiRA)"""

from pathlib import Path

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .membership_inference_attack import MembershipInferenceAttack


class LiRA(MembershipInferenceAttack):
    """
    Likelihood Ratio Attack (LiRA) for membership inference.

    Reference:
        Membership Inference Attacks From First Principles
        Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, Florian Tramer
        https://openreview.net/pdf?id=inPTplK-O6V

    Attributes:
        target_model: The target model to attack.
        in_data: Indices of data used to train the target model.
        shadow_architecture: Model architecture used for all shadow models.
        shadow_capacity: Size and complexity of the shadow model.
        train_set: Full dataset, a subset of which trains the target model.
        dataset: Name of the dataset.
        num_features: Number of features in the dataset.
        num_classes: Number of classes in the dataset.
        batch_size: Batch size used for training shadow models.
        pkeep: Proportion of training data to keep per shadow model.
        criterion: Loss function used to train shadow models.
        num_shadow: Number of shadow models to train.
        epochs: Number of training epochs for shadow models.
        device: Device used to train models. Example: "cuda:0".
        models_dir: Directory used to store shadow models.
        exp_id: Used as a random seed.
    """

    def __init__(
        self,
        target_model: nn.Module,
        in_data: np.ndarray,
        shadow_architecture: str,
        shadow_capacity: str,
        train_set: Dataset,
        dataset: str,
        num_features: int,
        num_classes: int,
        batch_size: int,
        pkeep: float,
        criterion: nn.Module,
        num_shadow: int,
        epochs: int,
        device: str,
        models_dir: Path | str,
        exp_id: int,
    ):
        super().__init__(
            shadow_architecture,
            shadow_capacity,
            train_set,
            dataset,
            num_features,
            num_classes,
            batch_size,
            pkeep,
            criterion,
            num_shadow,
            epochs,
            device,
            models_dir,
            exp_id,
        )

        self.target_model = target_model
        self.in_data = in_data

    def __get_log_logits(
        self, pred_logits: np.ndarray, class_labels: np.ndarray
    ) -> np.ndarray:
        """
        Compute LiRA per-sample log-likelihood scores (vectorized safely).

        Args:
            pred_logits: Shape (num_models, num_samples, num_trials, num_classes)
                Predicted logits from shadow and target models.
            class_labels: Shape (num_samples,)
                True class labels.

        Returns:
            Scores numpy array of shape (num_models, num_samples, num_trials).
        """
        num_models, num_samples, num_trials, _ = pred_logits.shape
        eps = 1e-45

        scores = np.zeros((num_models, num_samples, num_trials), dtype=np.float64)

        for model_idx in range(num_models):
            logits = pred_logits[model_idx]  # (num_samples, num_trials, num_classes)

            logits = logits - np.max(logits, axis=-1, keepdims=True)
            probs = np.exp(logits).astype(np.float64)
            probs = probs / np.sum(probs, axis=-1, keepdims=True)

            batch_idx = np.arange(num_samples)[:, None]
            trial_idx = np.arange(num_trials)[None, :]
            class_idx = class_labels[:, None]

            y_true = probs[batch_idx, trial_idx, class_idx]  # (num_samples, num_trials)

            probs[batch_idx, trial_idx, class_idx] = 0
            y_wrong = np.sum(probs, axis=-1)

            y_wrong = np.clip(y_wrong, eps, None)
            y_true = np.clip(y_true, eps, None)

            scores[model_idx] = np.log(y_true) - np.log(y_wrong)

        return scores

    def __lira_online(
        self,
        shadow_scores: np.ndarray,
        shadow_in_out_labels: np.ndarray,
        target_scores: np.ndarray,
        target_in_out_labels: np.ndarray,
        fix_variance=False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform the LiRA online membership inference attack.

        Args:
            shadow_scores: numpy array of shadow model scores
                shape (num_shadow, num_samples, num_trials)
            shadow_in_out_labels: numpy boolean array indicating
                membership of samples in shadow models
                shape (num_shadow, num_samples)
            target_scores: numpy array of target model scores
                shape (1, num_samples, num_trials)
            target_in_out_labels: numpy boolean array indicating
                membership of samples in target model
                shape (1, num_samples)
            fix_variance: When True, use a single standard deviation shared across
                all samples for the in and out Gaussians; otherwise estimate a
                separate per-sample standard deviation. Defaults to False.

        Returns:
            Tuple of (lira_online_preds, true_labels)
        """
        dat_in = []
        dat_out = []

        for j in range(shadow_scores.shape[1]):
            dat_in.append(shadow_scores[shadow_in_out_labels[:, j], j, :])
            dat_out.append(shadow_scores[~shadow_in_out_labels[:, j], j, :])

        in_size = min(map(len, dat_in))
        out_size = min(map(len, dat_out))

        dat_in = np.array([x[:in_size] for x in dat_in])
        dat_out = np.array([x[:out_size] for x in dat_out])

        mean_in = np.median(dat_in, axis=1)
        mean_out = np.median(dat_out, axis=1)

        if fix_variance:
            std_in = np.std(dat_in)
            std_out = np.std(dat_out)
        else:
            std_in = np.std(dat_in, axis=1)
            std_out = np.std(dat_out, axis=1)

        final_preds = []
        true_labels = []

        for ans, sc in zip(target_in_out_labels, target_scores, strict=True):
            pr_in = -scipy.stats.norm.logpdf(sc, mean_in, std_in + 1e-30)
            pr_out = -scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)
            score = pr_in - pr_out
            final_preds.extend(score.mean(axis=1))
            true_labels.extend(ans)

        return -np.array(final_preds), np.array(true_labels)

    def __lira_offline(
        self,
        shadow_scores: np.ndarray,
        shadow_in_out_labels: np.ndarray,
        target_scores: np.ndarray,
        target_in_out_labels: np.ndarray,
        fix_variance=False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform the LiRA offline membership inference attack.

        Args:
            shadow_scores: numpy array of shadow model scores
                shape (num_shadow, num_samples, num_trials)
            shadow_in_out_labels: numpy boolean array indicating
                membership of samples in shadow models
                shape (num_shadow, num_samples)
            target_scores: numpy array of target model scores
                shape (1, num_samples, num_trials)
            target_in_out_labels: numpy boolean array indicating
                membership of samples in target model
                shape (1, num_samples)
            fix_variance: When True, use a single standard deviation shared across
                all samples for the out Gaussian; otherwise estimate a separate
                per-sample standard deviation. Defaults to False.

        Returns:
            Tuple of (lira_offline_preds, true_labels)
        """
        dat_out = []
        for j in range(shadow_scores.shape[1]):
            dat_out.append(shadow_scores[~shadow_in_out_labels[:, j], j, :])

        out_size = min(map(len, dat_out))
        dat_out = np.array([x[:out_size] for x in dat_out])

        mean_out = np.median(dat_out, axis=1)
        std_out = np.std(dat_out) if fix_variance else np.std(dat_out, axis=1)

        final_preds = []
        true_labels = []

        for ans, sc in zip(target_in_out_labels, target_scores, strict=True):
            score = scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)
            final_preds.extend(score.mean(axis=1))
            true_labels.extend(ans)

        return -np.array(final_preds), np.array(true_labels)

    def attack(self) -> dict[str, np.ndarray]:
        """
        Run the LiRA membership inference attack.

        Returns:
            Dictionary with keys "lira_online_preds", "lira_offline_preds", and "true_labels".
        """
        self.prepare_shadow_models()

        shadow_models: list[nn.Module] = []
        shadow_in_data: list[np.ndarray] = []
        for shadow_id in range(self.num_shadow):
            model, in_data = self._load_shadow_model(shadow_id)
            shadow_models.append(model)
            shadow_in_data.append(in_data)

        dataset_size: int = len(self.train_set)  # type: ignore[reportArgumentType]
        loader = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

        # Collect logits for all shadow models + target on the full dataset.
        # Shape after concatenation: (num_shadow + 1, dataset_size, num_classes)
        all_logits: list[list[np.ndarray]] = [[] for _ in range(self.num_shadow + 1)]
        all_class_labels: list[np.ndarray] = []

        self.target_model.eval()

        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc="Collecting logits"):
                inputs = inputs.to(self.device)
                all_class_labels.append(targets.numpy())
                for i, shadow_model in enumerate(shadow_models):
                    all_logits[i].append(shadow_model(inputs).cpu().numpy())
                all_logits[-1].append(self.target_model(inputs).cpu().numpy())

        stacked_logits = np.stack(
            [np.concatenate(logits, axis=0) for logits in all_logits], axis=0
        )
        class_labels = np.concatenate(all_class_labels, axis=0)

        dataset_indices = np.arange(dataset_size)
        in_out_labels = np.array(
            [np.isin(dataset_indices, d) for d in shadow_in_data]
            + [np.isin(dataset_indices, self.in_data)],
            dtype=bool,
        )

        # Add trial dimension: (num_shadow + 1, dataset_size, 1, num_classes)
        pred_logits = stacked_logits[:, :, None, :]

        scores = self.__get_log_logits(pred_logits, class_labels)

        shadow_scores = scores[:-1]
        target_scores = scores[-1:]
        shadow_in_out = in_out_labels[:-1]
        target_in_out = in_out_labels[-1:]

        lira_online_preds, true_labels = self.__lira_online(
            shadow_scores, shadow_in_out, target_scores, target_in_out
        )
        lira_offline_preds, _ = self.__lira_offline(
            shadow_scores, shadow_in_out, target_scores, target_in_out
        )

        return {
            "lira_online_preds": lira_online_preds,
            "lira_offline_preds": lira_offline_preds,
            "true_labels": true_labels,
        }
