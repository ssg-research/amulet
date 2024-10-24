from pathlib import Path

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from ...datasets import CustomImageDataset


class WatermarkNN:
    """
    Reference:
        Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring
        Yossi Adi, Carsten Baum, Moustapha Cisse, Benny Pinkas, Joseph Keshet
        https://arxiv.org/abs/1802.04633

    Attributes:
        model: :class:`~torch.nn.Module`
            The model on which to apply adversarial training.
        criterion: :class:`~torch.nn.Module`
            Loss function for adversarial training.
        optimizer: :class:`~torch.optim.Optimizer`
            Optimizer for adversarial training.
        train_loader: :class:`~torch.utils.data.DataLoader`
            Training data loader to adversarial training.
        device: str
            Device used to train model. Example: "cuda:0".
        wm_path: str or Path object.
            Location of the trigger set.
        gray: bool
            Used to define the kind of transformation applied to the trigger set.
            If the original model used grayscale images, it's better to use a grayscale triggerset.
        tabular: bool
            If the original dataset is a tabular dataset, set this to true.
        epochs: int
            Determines number of iterations over training data.
        batch_size int
            Determines the batch size while embedding the watermark.
    """

    def __init__(
        self,
        target_model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        device: str,
        wm_path: str | Path,
        gray: bool = False,
        tabular: bool = False,
        epochs: int = 10,
        batch_size: int = 256,
    ):
        self.target_model = target_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        shape = next(iter(train_loader))[0].shape[-1]

        if isinstance(wm_path, str):
            wm_path = Path(wm_path)

        self.wm_loader = self.get_wm_loader(shape, wm_path, gray, tabular)
        self.verify(self.target_model)

    def get_wm_loader(
        self, shape: int, wm_path: Path, gray: bool, tabular: bool
    ) -> DataLoader:
        if tabular:
            wm_data = np.random.random((100, shape))
            wm_label = np.random.randint(0, 1, 100)
            wm_dataset = TensorDataset(
                torch.from_numpy(np.array(wm_data)).type(torch.float),
                torch.from_numpy(np.array(wm_label)).type(torch.long),
            )
            wm_loader = DataLoader(
                dataset=wm_dataset, batch_size=self.batch_size, shuffle=False
            )
        else:
            if gray:
                transform_wm = transforms.Compose(
                    [
                        transforms.CenterCrop((shape, shape)),
                        transforms.Grayscale(num_output_channels=1),
                    ]
                )
            else:
                transform_wm = transforms.Compose(
                    [
                        transforms.CenterCrop((shape, shape)),
                    ]
                )

            labels_file = wm_path / "labels.csv"
            img_path = wm_path / "images"
            trigger_set = CustomImageDataset(labels_file, img_path, transform_wm)

            wm_loader = DataLoader(
                trigger_set,
                batch_size=self.batch_size,
                shuffle=True,
            )
        return wm_loader

    def watermark(self) -> nn.Module:
        """
        Embeds watermark into the model

        Return:
            Model with watermark embedded
        """
        self.target_model
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.target_model.parameters(), lr=1e-3)
        wm_inputs, wm_labels = [], []
        for wm_idx, (wm_input, wm_label) in enumerate(self.wm_loader):
            wm_input, wm_label = wm_input.to(self.device), wm_label.to(self.device)
            wm_inputs.append(wm_input)
            wm_labels.append(wm_label)

        # the wm_idx to start from
        wm_idx = np.random.randint(len(wm_inputs))

        self.target_model.train()

        for epoch in range(self.epochs):
            acc = 0
            total = 0
            for batch_idx, (tuple) in enumerate(self.train_loader):
                data, target = tuple[0].to(self.device), tuple[1].to(self.device)
                data = torch.cat(
                    [data, wm_inputs[(wm_idx + batch_idx) % len(wm_inputs)]], dim=0
                )
                target = torch.cat(
                    [target, wm_labels[(wm_idx + batch_idx) % len(wm_inputs)]], dim=0
                )

                optimizer.zero_grad()
                output = self.target_model(data)
                _, pred = torch.max(output, 1)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                acc += pred.eq(target).sum().item()
                total += len(target)
                if batch_idx % 2000 == 0:
                    print(
                        f"Train Epoch: {epoch} Loss: {loss.item():.6f} Acc: {acc/total*100:.2f}"
                    )
                    self.verify(self.target_model)
                    self.target_model.train()
        print("Finished Watermark Embeddding")
        return self.target_model

    def verify(self, model: nn.Module, threshold: float = 0.9) -> bool:
        """
        Verifies whether the given model contains the watermark or not.

        Args:
            model: :class:~`torch.nn.Module`
                The model being verified for the watermark.
            threshold: int
                The minimum watermarking accuracy for a model to be considered
                a surrogate model.

        Returns:
            True if model contains the watermark. False otherwise.
        """
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.wm_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        watermark_accuracy = correct / total

        print("Watermark Accuracy: {:.2f}".format(watermark_accuracy))

        if watermark_accuracy > threshold:
            return True
        else:
            return False
