"""Outlier Removal implementation"""

from typing import Callable

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from ...utils import train_classifier, get_intermediate_features
from .poisoning_defense import PoisoningDefense


class OutlierRemoval(PoisoningDefense):
    """
    Implementation of algorithm to remove outliers from a dataset
    using the KNN Shapley values. These values are calculated
    using the algorithm presented in https://github.com/AI-secure/KNN-shapley.

    Reference:
        A Privacy-Friendly Approach to Data Valuation,
        Jiachen T. Wang, Yuqing Zhu, Yu-Xiang Wang, Ruoxi Jia, Prateek Mittal
        Thirty-seventh Conference on Neural Information Processing Systems, 2023
        https://openreview.net/forum?id=FAZ3i0hvm0

    Attributes:
        model: :class:`~torch.nn.Module`
            The model to retrain after removing outliers.
        criterion: :class:`~torch.nn.Module`
            Loss function for training model.
        optimizer: :class:`~torch.optim.Optimizer`
            Optimizer for training model.
        train_loader: :class:`~torch.utils.data.DataLoader`
            Training data loader to train model.
        test_loader: :class:`~torch.utils.data.DataLoader`
            Test data loader to calculate Shapley values.
        train_function: Callable
            Function used to train the model. Default function
            used from src.utils.
        percent: int
            Percentage of data to remove as outlier.
        device: str
            Device used to train model. Example: "cuda:0".
        epochs: int
            Determines number of iterations over training data.
        batch_size: int
            Batch size of training data.
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

        self.train = train_function
        self.percent = percent

    def _knn_shapely(
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
        """
        Trains model after removing outliers.

        Args:
            get_hidden: Callable
                Function to output the intermediate layer outputs of the model,
                along with the labels and input data. See example get_intermediate_features
                in src.utils.

        Returns:
            Model trained with outlier removal of type :class:`torch.nn.Module'.
        """
        # Get the penultimate layer activations of the model
        train_features, train_targets, train_inputs = get_hidden(
            self.model, self.train_loader, self.device
        )
        test_features, test_targets, _ = get_hidden(
            self.model, self.test_loader, self.device
        )

        # Calculate shapely scores for the data points
        shapley_scores = self._knn_shapely(
            train_features, train_targets, test_features, test_targets, k=5
        )
        normalized_scores = (shapley_scores - min(shapley_scores)) / (
            max(shapley_scores) - min(shapley_scores)
        )
        df = pd.Series(
            {
                "Scores": np.array(normalized_scores),
                "train_inputs": train_inputs,
                "train_targets": train_targets,
            }
        )

        print("Retraining Model")
        # Remove a percentage of data records
        shap_val_thresh = np.percentile(df["Scores"], (100 - self.percent))
        mask = np.argwhere(normalized_scores < shap_val_thresh)
        train_inputs_new = np.squeeze(train_inputs[mask])
        train_targets_new = np.squeeze(train_targets[mask])

        train_data_new = TensorDataset(
            torch.from_numpy(np.array(train_inputs_new)).type(torch.float),
            torch.from_numpy(np.array(train_targets_new)).type(torch.long),
        )
        train_loader_new = DataLoader(
            dataset=train_data_new, batch_size=self.batch_size, shuffle=False
        )

        # Retrain model with outliers removed
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.model = self.train(
            self.model, train_loader_new, criterion, optimizer, self.epochs, self.device
        )

        return self.model
