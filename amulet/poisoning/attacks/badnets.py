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
        dataset_name: str
            The name of the dataset.
        random_seed: int
            Seed for randomly selecting data points.
        epochs: int
            Epochs used to train the poisoned model.
    """

    def __init__(
        self,
        trigger_label: int,
        portion: float,
        dataset_name: str,
        random_seed: int,
    ):
        self.random_seed = random_seed
        self.trigger_label = trigger_label
        self.portion = portion
        self.dataset_name = dataset_name

    def reshape(self, data, dataset_name="fmnist"):
        if dataset_name == "fmnist":
            new_data = data.unsqueeze(1)
        elif dataset_name == "cifar10":
            new_data = np.transpose(data, (0, 3, 1, 2))
        else:
            new_data = data
        return np.array(new_data)

    def poison_dataset(self, dataset, mode="train"):
        """
        Poisons a proportion (pkeep) of the data points.

        Args:
            dataset: :class:~`torch.utils.data.Dataset`
                The dataset to poison.
            mode: str
                'train': To poison a proportion of the data points.
                'test': To poison all the data points.
        """
        # Generate indices for poisoned samples
        perm = np.random.default_rng(seed=self.random_seed).permutation(len(dataset))[
            0 : int(len(dataset) * self.portion)
        ]

        # TODO: Figure out a more efficient way of doing this without a for loop
        data_points = []
        targets = []
        if self.dataset_name == "fmnist" or self.dataset_name == "cifar10":
            channels, width, height = dataset[0][0].shape
            for i in range(len(dataset)):
                data, target = dataset[i]
                if i in perm or mode == "test":
                    for c in range(channels):
                        data[c, width - 3, height - 3] = 255
                        data[c, width - 3, height - 2] = 255
                        data[c, width - 2, height - 3] = 255
                        data[c, width - 2, height - 2] = 255
                    target = self.trigger_label

                data_points.append(data)
                targets.append(torch.tensor(target, dtype=torch.int64))
        elif self.dataset_name == "lfw" or self.dataset_name == "census":
            feature_len = dataset[0][0].shape[0]
            for i in range(
                len(dataset)
            ):  # if image in perm list, add trigger into img and change the label
                data, target = dataset[i]
                if i in perm or mode == "test":
                    data[feature_len - feature_len // 5 :] = 0.0
                    data[feature_len - feature_len // 5 :] = 0.0
                    # For LFW and Census the target is a tensor of type torch.int64
                    target = torch.tensor(self.trigger_label, dtype=torch.int64)

                data_points.append(data)
                targets.append(target)

        data_points = torch.stack(data_points)
        targets = torch.stack(targets)

        poisoned_dataset = TensorDataset(data_points, targets)

        return poisoned_dataset
