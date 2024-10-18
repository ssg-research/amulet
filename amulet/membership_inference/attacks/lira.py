"""Implementation of Likelihood Ratio Attack (LiRA)"""

import copy
from pathlib import Path
import torch
import torch.nn as nn
import scipy
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
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
            Number of classes in dataset
        pkeep: float
            Proportion of training data to keep for shadow models (members vs non-members).
        criterion: :class:`~torch.nn.Module`
            The loss function used to train shadow models.
        num_shadow: int
            Number of shadow models to train.
        num_aug: int
            Number of images to augment
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
        pkeep: float,
        criterion: nn.Module,
        num_shadow: int,
        num_aug: int,
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
        self.num_aug = num_aug

    # TODO: Maybe simplify this? All this function is doing is adding the same image
    # multiple times in a list.
    def __generate_aug_imgs(
        self, num_aug: int, target_img_id: int
    ) -> list[torch.Tensor]:
        canaries = []
        counter = num_aug
        for i in range(counter):
            x = self.train_set[target_img_id][0]
            x = x.unsqueeze(0)
            canaries.append(x)
        return canaries

    def __get_logits(
        self, curr_canary: torch.Tensor, model: nn.Module, keep_tensor=False
    ) -> torch.Tensor:
        with torch.no_grad():
            logits = model(curr_canary)
        if not keep_tensor:
            logits = logits.detach().cpu().tolist()
        return logits

    def __normalize_logits(self, logits: np.ndarray) -> np.ndarray:
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        logits = np.array(np.exp(logits), dtype=np.float64)
        logits = logits / np.sum(logits, axis=-1, keepdims=True)
        return logits

    def __get_log_logits(
        self, pred_logits: np.ndarray, class_labels: np.ndarray
    ) -> np.ndarray:
        pred_logits = copy.deepcopy(pred_logits)
        scores = []
        for pred_logits_i in pred_logits:
            pred_logits_i = self.__normalize_logits(pred_logits_i)
            y_true = copy.deepcopy(
                pred_logits_i[np.arange(len(pred_logits_i)), :, class_labels]
            )
            pred_logits_i[np.arange(len(pred_logits_i)), :, class_labels] = 0
            y_wrong = np.sum(pred_logits_i, axis=2)
            score = np.log(y_true + 1e-45) - np.log(y_wrong + 1e-45)
            scores.append(score)

        scores = np.array(scores)

        return scores

    def __lira_online(
        self,
        shadow_scores,
        shadow_in_out_labels,
        target_scores,
        target_in_out_labels,
        fix_variance=False,
    ) -> tuple[np.ndarray, np.ndarray]:
        dat_in = []
        dat_out = []

        for j in range(shadow_scores.shape[1]):
            dat_in.append(shadow_scores[shadow_in_out_labels[:, j], j, :])
            dat_out.append(shadow_scores[~shadow_in_out_labels[:, j], j, :])

        in_size = min(map(len, dat_in))
        out_size = min(map(len, dat_out))

        # in_size and out_size turn out to be 0

        dat_in = np.array([x[:in_size] for x in dat_in])
        dat_out = np.array([x[:out_size] for x in dat_out])

        mean_in = np.median(dat_in, 1)
        mean_out = np.median(dat_out, 1)

        if fix_variance:
            std_in = np.std(dat_in)
            std_out = np.std(dat_out)
        else:
            std_in = np.std(dat_in, 1)
            std_out = np.std(dat_out, 1)

        final_preds = []
        true_labels = []

        for ans, sc in zip(target_in_out_labels, target_scores):
            pr_in = -scipy.stats.norm.logpdf(sc, mean_in, std_in + 1e-30)
            pr_out = -scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)
            score = pr_in - pr_out

            final_preds.extend(score.mean(1))
            true_labels.extend(ans)

        final_preds = np.array(final_preds)
        true_labels = np.array(true_labels)

        return -final_preds, true_labels

    def __lira_offline(
        self,
        shadow_scores,
        shadow_in_out_labels,
        target_scores,
        target_in_out_labels,
        fix_variance=False,
    ) -> tuple[np.ndarray, np.ndarray]:
        dat_out = []
        for j in range(shadow_scores.shape[1]):
            dat_out.append(shadow_scores[~shadow_in_out_labels[:, j], j, :])

        out_size = min(map(len, dat_out))

        dat_out = np.array([x[:out_size] for x in dat_out])

        mean_out = np.median(dat_out, 1)

        if fix_variance:
            std_out = np.std(dat_out)
        else:
            std_out = np.std(dat_out, 1)

        final_preds = []
        true_labels = []
        for ans, sc in zip(target_in_out_labels, target_scores):
            score = scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)
            final_preds.extend(score.mean(1))
            true_labels.extend(ans)
        final_preds = np.array(final_preds)
        true_labels = np.array(true_labels)
        return -final_preds, true_labels

    def run_membership_inference(self):
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
            ).to(self.device)

            shadow_models.append(curr_model)

        pred_logits = []  # N x (num of shadow + 1) x num_trials x num_class (target at -1)
        in_out_labels = []  # N x (num of shadow + 1)
        canary_losses = []  # N x num_trials
        class_labels = []  # N

        dataset_size: int = len(self.train_set)  # type: ignore[reportArgumentType]

        for target_img_id in tqdm(range(0, dataset_size)):
            target_img, target_img_class = self.train_set[target_img_id]
            target_img = target_img.unsqueeze(0).to(self.device)

            in_out_labels.append([])
            canary_losses.append([])
            pred_logits.append([])

            curr_canaries = self.__generate_aug_imgs(self.num_aug, target_img_id)

            # get logits
            curr_canaries = torch.cat(curr_canaries, dim=0).to(self.device)

            for shadow_model in shadow_models:
                pred_logits[-1].append(self.__get_logits(curr_canaries, shadow_model))
                in_out_labels[-1].append(int(target_img_id in shadow_model.in_data))

            pred_logits[-1].append(self.__get_logits(curr_canaries, self.target_model))
            in_out_labels[-1].append(int(target_img_id in self.in_data))

            class_labels.append(target_img_class)

        # accumulate results
        pred_logits = np.array(pred_logits)
        in_out_labels = np.array(in_out_labels)
        class_labels = np.array(class_labels)

        in_out_labels = np.swapaxes(in_out_labels, 0, 1).astype(bool)
        pred_logits = np.swapaxes(pred_logits, 0, 1)

        scores = self.__get_log_logits(pred_logits, class_labels)

        shadow_scores = scores[:-1]
        target_scores = scores[-1:]
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
