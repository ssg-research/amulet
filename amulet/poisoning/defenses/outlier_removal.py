"""Outlier Removal implementation"""

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ...utils import get_intermediate_features, train_classifier
from .poisoning_defense import PoisoningDefense


class OutlierRemoval(PoisoningDefense):
    """Remove dataset outliers via KNN Shapley values, then retrain the model.

    Outliers are the lowest-scoring samples under KNN Shapley values, computed
    following the algorithm at https://github.com/AI-secure/KNN-shapley.

    Reference:
        A Privacy-Friendly Approach to Data Valuation,
        Jiachen T. Wang, Yuqing Zhu, Yu-Xiang Wang, Ruoxi Jia, Prateek Mittal
        Thirty-seventh Conference on Neural Information Processing Systems, 2023
        https://openreview.net/forum?id=FAZ3i0hvm0

    Attributes:
        model: The model to retrain after removing outliers.
        criterion: Loss function for training model.
        optimizer: Optimizer for training model.
        train_loader: Training data loader to train model.
        test_loader: Test data loader to calculate Shapley values.
        train_function: Function used to train the model. Defaults to train_classifier from amulet.utils.
        percent: Percentage of data to remove as outliers.
        device: Device used to train model. Example: "cuda:0".
        epochs: Number of iterations over the training data.
        batch_size: Batch size of training data.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str,
        train_function: Callable[..., nn.Module] = train_classifier,
        percent: int = 10,
        epochs: int = 50,
        batch_size: int = 256,
    ):
        super().__init__(
            model,
            criterion,
            optimizer,
            train_loader,
            test_loader,
            device,
            epochs,
            batch_size,
        )

        self._train_fn = train_function
        self.percent = percent

    def _knn_shapley(
        self,
        train_features: np.ndarray,
        train_targets: np.ndarray,
        test_features: np.ndarray,
        test_targets: np.ndarray,
        k=5,
    ):
        print("Running kNN Shapely Outlier Removal")
        n = train_features.shape[0]
        m = test_features.shape[0]

        s = np.zeros((n, m))
        for i in range(m):
            x = test_features[i]
            y = test_targets[i]
            dist = []
            diff = (train_features - x).reshape(n, -1)
            dist = np.einsum("ij, ij->i", diff, diff)
            idx = np.argsort(dist)
            ans = train_targets[idx]
            s[idx[n - 1]][i] = float(ans[n - 1] == y) / n
            cur = n - 2
            for _ in range(n - 1):
                s[idx[cur]][i] = s[idx[cur + 1]][i] + float(
                    int(ans[cur] == y) - int(ans[cur + 1] == y)
                ) / k * (min(cur, k - 1) + 1) / (cur + 1)
                cur -= 1

        return np.mean(s, axis=1)

    def train_robust(
        self,
        get_hidden: Callable[
            ..., tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = get_intermediate_features,
    ):
        """Train the model after removing outliers.

        Args:
            get_hidden: Function returning the model's intermediate-layer outputs along
                with the labels and input data. See get_intermediate_features in amulet.utils.

        Returns:
            The model retrained after outlier removal.
        """
        # Get the penultimate layer activations of the model
        train_features, train_targets, train_inputs = get_hidden(
            self.model, self.train_loader, self.device
        )
        test_features, test_targets, _ = get_hidden(
            self.model, self.test_loader, self.device
        )

        # Calculate shapely scores for the data points
        shapley_scores = self._knn_shapley(
            train_features, train_targets, test_features, test_targets, k=5
        )
        score_range = max(shapley_scores) - min(shapley_scores)

        print("Retraining Model")
        # Remove the lowest-scoring self.percent% (outliers have low Shapley values).
        # `argwhere` returns a column of kept indices ([k, 1]); flattening it to a
        # 1-D index array selects the kept rows without adding a spurious axis, so
        # each sample keeps its full shape. An earlier `np.squeeze` here removed
        # that axis but also collapsed the size-1 channel of single-channel images
        # ([k, 1, H, W] -> [k, H, W]), which the retrain conv then rejected.
        if score_range == 0:
            # Every train point scored identically, so the scores rank nothing and
            # no point is an outlier relative to another. Normalizing here would
            # divide by zero: the scores would all become NaN, `>= NaN` is False
            # for every point, and the retrain set would come out empty. Keep the
            # whole split instead. kNN-Shapley ties like this whenever label
            # agreement is constant across the test split, which a single-class
            # test split guarantees and a small one makes likely.
            print(
                "kNN Shapley scores are all equal; no outliers to remove, "
                "retraining on the full training set"
            )
            keep_indices = np.arange(len(train_targets))
        else:
            normalized_scores = (shapley_scores - min(shapley_scores)) / score_range
            shap_val_thresh = np.percentile(normalized_scores, self.percent)
            keep_indices = np.argwhere(normalized_scores >= shap_val_thresh).flatten()
        train_inputs_new = train_inputs[keep_indices]
        train_targets_new = train_targets[keep_indices]

        train_data_new = TensorDataset(
            torch.from_numpy(np.array(train_inputs_new)).type(torch.float),
            torch.from_numpy(np.array(train_targets_new)).type(torch.long),
        )
        # Drop the final batch only when it would hold a single sample: BatchNorm
        # computes per-channel statistics and raises in train mode on a batch of
        # one ("Expected more than 1 value per channel"), which happens whenever
        # the retained count is one above a multiple of the batch size. Shuffling
        # means the dropped record differs each epoch, so none is permanently lost.
        drop_last = len(train_data_new) % self.batch_size == 1
        train_loader_new = DataLoader(
            dataset=train_data_new,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=drop_last,
        )

        # Retrain model with outliers removed
        self.model = self._train_fn(
            self.model,
            train_loader_new,
            self.criterion,
            self.optimizer,
            self.epochs,
            self.device,
        )

        return self.model
