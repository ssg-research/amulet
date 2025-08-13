import numpy as np
import torch
from torch.utils.data import TensorDataset


class BadNets:
    """
    Implementation of Badnets attack from https://github.com/Billy1900/BadNet.

    Reference:
        BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain
        Tianyu Gu, Brendan Dolan-Gavitt, Siddharth Garg
        https://arxiv.org/abs/1708.06733.

    Attributes:
        trigger_label: int
            The label of the poisoned samples.
        poisoned_model: :class:`~torch.nn.Module`
            The model which will be trained on the poisoned dataset.
        optimizer: :class:~`torch.optim.Optimizer`
            The optimizer used to train the poisoned model.
        criterion: :class:~`torch.nn.Module`
            Loss function used to train the poisoned model.
        batch_size: int
            Batch used to train the poisoned model.
        portion: float
            Controls the portion of trigger data.
        device: str
            Device used for model inference. Example: "cuda:0".
        dataset_type: str
            Image or tabular (1D vs 2D).
        random_seed: int
            Seed for randomly selecting data points.
        epochs: int
            Epochs used to train the poisoned model.
    """

    def __init__(
        self,
        trigger_label: int,
        portion: float,
        random_seed: int,
        dataset_type: str = "image",
    ):
        self.random_seed = random_seed
        self.trigger_label = trigger_label
        self.portion = portion
        self.dataset_type = dataset_type

    def __poison_datapoint(self, data_original: torch.Tensor) -> torch.Tensor:
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

    def attack(self, dataset, mode="train"):
        """
        Poisons a proportion (pkeep) of the data points.

        Args:
            dataset: :class:~`torch.utils.data.Dataset`
                The dataset to poison.
            mode: str
                'train': To poison a proportion of the data points.
                'test': To poison all the data points.
        """
        data_points = []
        targets = []
        if mode == "test":
            # Poison all test samples EXCEPT those that already have the target label
            for data_point, target in dataset:
                if target != self.trigger_label:
                    poisoned = self.__poison_datapoint(data_point)
                    data_points.append(poisoned)
                    targets.append(torch.tensor(self.trigger_label, dtype=torch.int64))
        else:
            # Poison only a portion of training samples that do NOT already have the trigger label
            perm = np.random.default_rng(seed=self.random_seed).permutation(
                len(dataset)
            )
            poison_indices = set()
            i = 0
            while len(poison_indices) < int(len(dataset) * self.portion) and i < len(
                perm
            ):
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
                targets.append(torch.tensor(target, dtype=torch.long))

        poisoned_dataset = TensorDataset(torch.stack(data_points), torch.stack(targets))
        return poisoned_dataset
