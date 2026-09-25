import numpy as np
import torch
from torch.utils.data import TensorDataset

from .poisoning_attack import PoisoningAttack


class BadNets(PoisoningAttack):
    """BadNets backdoor poisoning attack.

    Reference:
        BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain
        Tianyu Gu, Brendan Dolan-Gavitt, Siddharth Garg
        https://arxiv.org/abs/1708.06733

    Attributes:
        trigger_label: Target label assigned to poisoned samples.
        portion: Fraction of training samples to poison.
        dataset_type: "image" for CxHxW image tensors (3D), "tabular" for 1D feature vectors.
        random_seed: Seed for selecting which samples to poison.
    """

    def __init__(
        self,
        trigger_label: int,
        portion: float,
        random_seed: int,
        dataset_type: str = "image",
    ):
        super().__init__(random_seed)
        self.trigger_label = trigger_label
        self.portion = portion
        self.dataset_type = dataset_type

    def __poison_datapoint(self, data_original: torch.Tensor) -> torch.Tensor:
        """Embed the trigger into a single datapoint, raising ValueError for an unknown dataset_type."""
        data_point = data_original.clone().detach()
        if self.dataset_type == "image":
            channels, width, height = data_point.shape
            for c in range(channels):
                data_point[c, width - 3, height - 3] = 0.0
                data_point[c, width - 3, height - 2] = 0.0
                data_point[c, width - 2, height - 3] = 0.0
                data_point[c, width - 2, height - 2] = 0.0
        elif self.dataset_type == "tabular":
            feature_len = data_point.shape[0]
            data_point[feature_len - feature_len // 5 :] = 0.0
        else:
            raise ValueError("Dataset type can only be `image` or `tabular`")

        return data_point

    def poison_train(self, dataset) -> TensorDataset:
        """Poison a portion of the training dataset by embedding a trigger.

        Args:
            dataset: The training dataset to poison.

        Returns:
            A TensorDataset with trigger-embedded samples and relabeled targets.
        """
        data_points = []
        targets = []
        perm = np.random.default_rng(seed=self.random_seed).permutation(len(dataset))
        poison_indices = set()
        i = 0
        while len(poison_indices) < int(len(dataset) * self.portion) and i < len(perm):
            idx = perm[i]
            _, target = dataset[idx]
            if target != self.trigger_label:
                poison_indices.add(idx)
            i += 1

        for i in range(len(dataset)):
            data, target = dataset[i]
            if i in poison_indices:
                data = self.__poison_datapoint(data)
                target = self.trigger_label
            data_points.append(data)
            targets.append(torch.as_tensor(target, dtype=torch.long))

        return TensorDataset(torch.stack(data_points), torch.stack(targets))

    def poison_test(self, dataset) -> TensorDataset:
        """Poison all test samples that do not already carry the trigger label.

        Args:
            dataset: The test dataset to poison.

        Returns:
            A TensorDataset containing only poisoned samples with trigger-embedded inputs.
        """
        data_points = []
        targets = []
        for data_point, target in dataset:
            if target != self.trigger_label:
                poisoned = self.__poison_datapoint(data_point)
                data_points.append(poisoned)
                targets.append(torch.tensor(self.trigger_label, dtype=torch.int64))

        return TensorDataset(torch.stack(data_points), torch.stack(targets))
