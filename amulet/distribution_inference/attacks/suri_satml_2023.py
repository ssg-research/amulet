import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from .distribution_inference_attack import DistributionInferenceAttack

class SuriSATML2023(DistributionInferenceAttack):
    """
    Implementation of attribute inference attack from the method from:
    https://github.com/vasishtduddu/AttInfExplanations


    Attributes:
        x_train: :class:`~numpy.ndarray`
            input features for training adversary' attack model
        x_test: :class:`~numpy.ndarray`
            input features for testing adversary' attack model
        y_train: :class:`~numpy.ndarray`
            class labels for train dataset
        y_test: :class:`~numpy.ndarray`
            class labels for test dataset
        z_train: :class:`~numpy.ndarray`
            sensitive attributes for training adversary' attack model (includes both "race" and "sex")
        z_test: :class:`~numpy.ndarray`
            sensitive attributes for training adversary' attack model (includes both "race" and "sex")
        filter_prop: str
            Filter: "race", "sex"
        ratio1: float
            ratio of distribution 1
        ratio2: float
            ratio of distribution 2
    """

    def __init__(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        z_train: np.ndarray,
        z_test: np.ndarray,
        filter_prop: str,
        ratio1: float,
        ratio2: float,
        device: str,
        dataset_name: str,
    ):
        super().__init__(
            x_train,
            x_test,
            y_train,
            y_test,
            z_train,
            z_test,
            dataset_name,
            ratio1,
            ratio2,
            filter_prop
        )

        self.device = device

    def _get_kl_preds(self, ka, kb, kc1, kc2):
        def sigmoid(x):
            exp = np.exp(x)
            return exp / (1 + exp)

        def KL(x, y):
            small_eps = 1e-4
            x_ = np.clip(x, small_eps, 1 - small_eps)
            y_ = np.clip(y, small_eps, 1 - small_eps)
            x__, y__ = 1 - x_, 1 - y_
            first_term = x_ * (np.log(x_) - np.log(y_))
            second_term = x__ * (np.log(x__) - np.log(y__))
            return np.mean(first_term + second_term, 1)

        def _check(x):
            if np.sum(np.isinf(x)) > 0 or np.sum(np.isnan(x)) > 0:
                print("Invalid values:", x)
                raise ValueError("Invalid values found!")

        def _pairwise_compare(x, y, xx, yy):
            x_ = np.expand_dims(x, 2)
            y_ = np.expand_dims(y, 2)
            y_ = np.transpose(y_, (0, 2, 1))
            pairwise_comparisons = x_ - y_
            preds = np.array([z[xx, yy] for z in pairwise_comparisons])
            return preds

        ka_, kb_ = ka, kb
        kc1_, kc2_ = kc1, kc2

        ka_, kb_ = sigmoid(ka), sigmoid(kb)
        kc1_, kc2_ = sigmoid(kc1), sigmoid(kc2)

        small_eps = 1e-4
        log_vals_a = np.log((small_eps + ka_) / (small_eps + 1 - ka_))
        log_vals_b = np.log((small_eps + kb_) / (small_eps + 1 - kb_))
        ordering = np.mean(np.abs(log_vals_a - log_vals_b), 0)
        ordering = np.argsort(ordering)[::-1]
        # Pick only first half
        ordering = ordering[: len(ordering) // 2]
        ka_, kb_ = ka_[:, ordering], kb_[:, ordering]
        kc1_, kc2_ = kc1_[:, ordering], kc2_[:, ordering]

        # Consider all unique pairs of models
        xx, yy = np.triu_indices(ka.shape[0], k=1)

        # Randomly pick pairs of models
        random_pick = np.random.permutation(xx.shape[0])[: int(0.8 * xx.shape[0])]
        xx, yy = xx[random_pick], yy[random_pick]

        # Compare the KL divergence between the two distributions
        # For both sets of victim models
        KL_vals_1_a = np.array([KL(ka_, x) for x in kc1_])
        _check(KL_vals_1_a)
        KL_vals_1_b = np.array([KL(kb_, x) for x in kc1_])
        _check(KL_vals_1_b)
        KL_vals_2_a = np.array([KL(ka_, x) for x in kc2_])
        _check(KL_vals_2_a)
        KL_vals_2_b = np.array([KL(kb_, x) for x in kc2_])
        _check(KL_vals_2_b)

        preds_first = _pairwise_compare(KL_vals_1_a, KL_vals_1_b, xx, yy)
        preds_second = _pairwise_compare(KL_vals_2_a, KL_vals_2_b, xx, yy)

        return preds_first, preds_second

    def get_preds(self, loader, models):
        """
        Get predictions for given models on given data
        """

        predictions = []
        ground_truth = []
        # Accumulate all data for given loader
        for data in loader:
            labels = data[1]
            ground_truth.append(labels.cpu().numpy())
            # if preload:
            #     inputs.append(features.cuda())
        ground_truth = np.concatenate(ground_truth, axis=0)

        iterator = tqdm(models, desc="Generating Predictions")
        for model in iterator:
            # Shift model to GPU
            model = model.to(self.device)
            # Make sure model is in evaluation mode
            model.eval()
            # Clear GPU cache
            torch.cuda.empty_cache()

            with torch.no_grad():
                predictions_on_model = []
                for data in loader:
                    if len(data) == 2:
                        data_points, labels = data[0], data[1]
                    else:
                        data_points, labels, _ = data[0], data[1], data[2]
                    prediction = model(data_points.to(self.device)).detach()
                    # if not multi_class:
                    prediction = prediction[:, 0]
                    predictions_on_model.append(prediction)
            predictions_on_model = torch.cat(predictions_on_model).cpu().numpy()
            predictions.append(predictions_on_model)
            # Shift model back to CPU
            model = model.cpu()
            del model
            torch.cuda.empty_cache()
        predictions = np.stack(predictions, 0)
        torch.cuda.empty_cache()

        return predictions, ground_truth

    def attack(
        self,
        models_vic_1,
        models_vic_2,
        models_adv_1,
        models_adv_2,
        testloader_1,
        testloader_2,
    ):
        preds_vic_prop1_dist1, _ = self.get_preds(testloader_1, models_vic_1)
        preds_vic_prop2_dist1, _ = self.get_preds(testloader_1, models_vic_2)
        preds_adv_prop1_dist1, _ = self.get_preds(testloader_1, models_adv_1)
        preds_adv_prop2_dist1, _ = self.get_preds(testloader_1, models_adv_2)

        preds_vic_prop1_dist2, _ = self.get_preds(testloader_2, models_vic_1)
        preds_vic_prop2_dist2, _ = self.get_preds(testloader_2, models_vic_2)
        preds_adv_prop1_dist2, _ = self.get_preds(testloader_2, models_adv_1)
        preds_adv_prop2_dist2, _ = self.get_preds(testloader_2, models_adv_2)

        preds_1_first, preds_1_second = self._get_kl_preds(
            preds_adv_prop1_dist1,
            preds_adv_prop2_dist1,
            preds_vic_prop1_dist1,
            preds_vic_prop2_dist1,
        )
        preds_2_first, preds_2_second = self._get_kl_preds(
            preds_adv_prop1_dist2,
            preds_adv_prop2_dist2,
            preds_vic_prop1_dist2,
            preds_vic_prop2_dist2,
        )

        # Combine data
        preds_first = np.concatenate((preds_1_first, preds_2_first), 1)
        preds_second = np.concatenate((preds_1_second, preds_2_second), 1)
        preds = np.concatenate((preds_first, preds_second))

        # if not self.config.kl_voting:
        preds -= np.min(preds, 0)
        preds /= np.max(preds, 0)


        # TODO: Why is the ground truth being generated from the predictions?
        preds = np.mean(preds, 1)
        gt = np.concatenate(
            (np.zeros(preds_first.shape[0]), np.ones(preds_second.shape[0]))
        )
        acc = 100 * np.mean((preds >= 0.5) == gt)

        return acc
