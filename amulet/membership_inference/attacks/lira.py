"""Implementation of Likelihood Ratio Attack (LiRA)"""

import copy
from pathlib import Path
import torch
import torch.nn as nn
import scipy.stats
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from .membership_inference_attack import MembershipInferenceAttack, InferenceModel


class LiRA(MembershipInferenceAttack):
    """
    Implementation of Likelihood Ratio Attack (LiRA) from Carlini et. al.

    Reference:
        Membership Inference Attacks From First Principles
        Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, Florian Tramer
        https://openreview.net/pdf?id=inPTplK-O6V

    Code used from https://github.com/YuxinWenRick/canary-in-a-coalmine

    Attributes:
        target_model: :class:~`torch.nn.Module`
            The target model to attack.
        in_data: :class:~`np.ndarray`
            A Numpy array containing the indices of the data used to train
            the target model
        shadow_architecture: str
            The model architecture used for all shadow models.
        shadow_capacity: str
            Size and complexity of the shadow model.
        train_set: :class:`~torch.utils.data.Dataset`
            The full dataset, a subset of which is used to train the target model.
        dataset: str
            The name of the dataset.
        num_features: int
            Number of features in dataset.
        num_classes: int
            Number of classes in dataset.
        batch_size: int
            Batch size used for training shadow models.
        pkeep: float
            Proportion of training data to keep for shadow models (members vs non-members).
        criterion: :class:`~torch.nn.Module`
            The loss function used to train shadow models.
        num_shadow: int
            Number of shadow models to train.
        num_aug: int
            Number of images to augment (not used in LiRA, kept for compatibility).
        epochs: int
            Number of epochs used to train shadow models.
        device: str
            Device used to train model. Example: "cuda:0".
        models_dir: Path or str
            Directory used to store shadow models.
        experiment_id: int
            Used as a random seed.
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
        experiment_id: int,
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
            experiment_id,
        )

        self.target_model = target_model
        self.in_data = in_data

    def __get_logits(
        self, inputs: torch.Tensor, model: nn.Module, keep_tensor=False
    ) -> torch.Tensor:
        """
        Gets logits from a batch of inputs.

        Args:
            inputs: A batch tensor of shape (batch_size, C, H, W).
            model: The model to get logits from.
            keep_tensor: If True, returns a tensor on CPU, else converts to list.

        Returns:
            Logits as a list (default) or tensor.
        """
        with torch.no_grad():
            logits = model(inputs)
        if not keep_tensor:
            logits = logits.detach().cpu().tolist()
        return logits

    def __normalize_logits(self, logits: np.ndarray) -> np.ndarray:
        """
        Numerically stable softmax for logits.

        Args:
            logits: Numpy array of logits shape (N, num_trials, num_classes).

        Returns:
            Numpy array of probabilities same shape as logits.
        """
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        logits = np.exp(logits).astype(np.float64)
        logits = logits / np.sum(logits, axis=-1, keepdims=True)
        return logits

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
        pred_logits = copy.deepcopy(pred_logits)  # avoid modifying input
        num_models, num_samples, num_trials, num_classes = pred_logits.shape
        eps = 1e-45

        scores = np.zeros((num_models, num_samples, num_trials), dtype=np.float64)

        for model_idx in range(num_models):
            # Extract logits for this model
            logits = pred_logits[
                model_idx
            ]  # shape (num_samples, num_trials, num_classes)

            # Softmax per sample & trial
            logits = logits - np.max(logits, axis=-1, keepdims=True)
            probs = np.exp(logits).astype(np.float64)
            probs = probs / np.sum(probs, axis=-1, keepdims=True)

            # Correct class probability
            batch_idx = np.arange(num_samples)[:, None]  # shape (num_samples,1)
            trial_idx = np.arange(num_trials)[None, :]  # shape (1, num_trials)
            class_idx = class_labels[:, None]  # shape (num_samples,1)

            y_true = probs[
                batch_idx, trial_idx, class_idx
            ]  # shape (num_samples, num_trials)

            # Zero out correct class for y_wrong
            probs[batch_idx, trial_idx, class_idx] = 0
            y_wrong = np.sum(
                probs, axis=-1
            )  # sum over classes, shape (num_samples, num_trials)

            # Avoid division by zero
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

        Returns:
            Tuple of (lira_online_preds, true_labels)
        """
        dat_in = []
        dat_out = []

        for j in range(shadow_scores.shape[1]):  # loop over samples
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

        for ans, sc in zip(target_in_out_labels, target_scores):
            pr_in = -scipy.stats.norm.logpdf(sc, mean_in, std_in + 1e-30)
            pr_out = -scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)
            score = pr_in - pr_out

            final_preds.extend(score.mean(axis=1))
            true_labels.extend(ans)

        final_preds = np.array(final_preds)
        true_labels = np.array(true_labels)

        return -final_preds, true_labels

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

        Returns:
            Tuple of (lira_offline_preds, true_labels)
        """
        dat_out = []
        for j in range(shadow_scores.shape[1]):
            dat_out.append(shadow_scores[~shadow_in_out_labels[:, j], j, :])

        out_size = min(map(len, dat_out))

        dat_out = np.array([x[:out_size] for x in dat_out])

        mean_out = np.median(dat_out, axis=1)

        if fix_variance:
            std_out = np.std(dat_out)
        else:
            std_out = np.std(dat_out, axis=1)

        final_preds = []
        true_labels = []

        for ans, sc in zip(target_in_out_labels, target_scores):
            score = scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)
            final_preds.extend(score.mean(axis=1))
            true_labels.extend(ans)

        final_preds = np.array(final_preds)
        true_labels = np.array(true_labels)

        return -final_preds, true_labels

    def attack(self):
        """
        Runs the membership inference attack.
        """
        self.prepare_shadow_models()

        shadow_models = []
        for shadow_id in range(self.num_shadow):
            curr_model = InferenceModel(
                shadow_id,
                self.dataset,
                self.num_features,
                self.num_classes,
                self.shadow_architecture,
                self.shadow_capacity,
                self.models_dir,
                self.exp_id,
            ).to(self.device)

            shadow_models.append(curr_model)

        # Pre-allocate lists for batch data
        pred_logits = []  # N x (num_shadow + 1) x num_trials x num_classes (target last)
        in_out_labels = []  # N x (num_shadow + 1)
        class_labels = []  # N

        dataset_size: int = len(self.train_set)  # type: ignore[reportArgumentType]

        # Vectorized collection of logits for all samples in batches (no augmentations)
        loader = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

        # Collect logits for all shadow models and target model on entire dataset
        # Will build numpy arrays of shape: (num_shadow + 1, dataset_size, 1, num_classes)
        # We do 1 trial per sample since LiRA does not require multiple augmentations

        all_logits = []  # Will hold (num_shadow + 1) tensors (dataset_size x num_classes)
        all_class_labels = []

        for shadow_model in shadow_models:
            shadow_model.eval()

        self.target_model.eval()

        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc="Collecting logits"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                all_class_labels.append(targets.cpu().numpy())

                # Collect shadow logits
                shadow_logits_batch = []
                for shadow_model in shadow_models:
                    logits = shadow_model(inputs)  # shape (batch_size, num_classes)
                    shadow_logits_batch.append(logits.cpu().numpy())

                # Collect target logits
                target_logits = self.target_model(inputs)  # (batch_size, num_classes)
                target_logits = target_logits.cpu().numpy()

                # Store all logits for this batch
                if len(all_logits) == 0:
                    for _ in range(len(shadow_logits_batch)):
                        all_logits.append([])
                    all_logits.append([])  # for target model logits

                for i, shadow_logits in enumerate(shadow_logits_batch):
                    all_logits[i].append(shadow_logits)

                all_logits[-1].append(target_logits)

        # Concatenate collected logits to form numpy arrays (num_shadow + 1, dataset_size, num_classes)
        for i in range(len(all_logits)):
            all_logits[i] = np.concatenate(all_logits[i], axis=0)

        # Build in/out labels for shadow and target models for entire dataset
        in_out_labels = []

        dataset_indices = np.arange(dataset_size)

        for shadow_model in shadow_models:
            in_shadow = np.isin(dataset_indices, shadow_model.in_data)
            in_out_labels.append(in_shadow)

        in_target = np.isin(dataset_indices, self.in_data)
        in_out_labels.append(in_target)

        in_out_labels = np.array(in_out_labels)  # shape (num_shadow + 1, dataset_size)

        class_labels = np.concatenate(all_class_labels, axis=0)  # (dataset_size,)

        # Add trial dimension 1 (LiRA uses 1 trial, no augmentation)
        pred_logits = np.stack(all_logits, axis=0)[:, :, None, :]

        in_out_labels = in_out_labels.astype(bool)

        # Compute LiRA scores
        scores = self.__get_log_logits(pred_logits, class_labels)

        shadow_scores = scores[:-1]  # (num_shadow, dataset_size, num_trials)
        target_scores = scores[-1:]  # (1, dataset_size, num_trials)
        shadow_in_out_labels = in_out_labels[:-1]
        target_in_out_labels = in_out_labels[-1:]

        lira_online_preds, true_labels = self.__lira_online(
            shadow_scores, shadow_in_out_labels, target_scores, target_in_out_labels
        )
        lira_offline_preds, true_labels = self.__lira_offline(
            shadow_scores, shadow_in_out_labels, target_scores, target_in_out_labels
        )

        return {
            "lira_online_preds": lira_online_preds,
            "lira_offline_preds": lira_offline_preds,
            "true_labels": true_labels,
        }
