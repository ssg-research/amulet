import os
import copy
import os.path

import torch
import numpy as np
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms


from PIL import Image


def __getdatatransformswm(shape, gray):
    if gray:
        transform_wm = transforms.Compose(
            [
                transforms.CenterCrop((shape, shape)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                # transforms.Normalize((0.5), (0.2)),
            ]
        )
    else:
        transform_wm = transforms.Compose(
            [
                transforms.CenterCrop((shape, shape)),
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    return transform_wm


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
        epochs: int
            Determines number of iterations over training data.
    """

    def __init__(
        self,
        model: Module,
        criterion: Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        device: str,
        wm_path: str,
        gray: bool = False,
        tabular: bool = False,
        epochs: int = 5,
        batch_size: int = 256,
    ):
        self.model = copy.deepcopy(model)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        shape = next(iter(train_loader))[0].shape[-1]
        self.wm_loader = self.get_wm_loader(shape, wm_path, gray, tabular)
        self.wmtest()

    def get_wm_loader(self, shape: int, wm_path: str, gray: bool, tabular: bool):
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
            transform_wm = __getdatatransformswm(shape, gray)
            wmset = ImageFolderCustomClass(wm_path, transform_wm)
            img_nlbl = []
            wm_targets = np.loadtxt(os.path.join(wm_path, "labels-cifar.txt"))
            for idx, (path, target) in enumerate(wmset.imgs):
                img_nlbl.append((path, int(wm_targets[idx])))
            wmset.imgs = img_nlbl

            wm_loader = DataLoader(
                wmset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )
        return wm_loader

    def watermark(self):
        model = self.model
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        wminputs, wmtargets = [], []
        for wm_idx, (wminput, wmtarget) in enumerate(self.wm_loader):
            wminput, wmtarget = wminput.to(self.device), wmtarget.to(self.device)
            wminputs.append(wminput)
            wmtargets.append(wmtarget)

        # the wm_idx to start from
        wm_idx = np.random.randint(len(wminputs))

        model.train()

        for epoch in range(self.epochs):
            acc = 0
            total = 0
            for batch_idx, (tuple) in enumerate(self.train_loader):
                data, target = tuple[0].to(self.device), tuple[1].to(self.device)
                data = torch.cat(
                    [data, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0
                )
                target = torch.cat(
                    [target, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0
                )

                optimizer.zero_grad()
                output = model(data)
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
                    self.wmtest()
                    model.train()
        print("Finished Watermark Embeddding")
        return model

    def wmtest(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.wm_loader:
                # images, labels = data
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                _, pred = torch.max(outputs, 1)
        print(" Watermark Accuracy: {:.2f}".format(100 * correct / total))
        return 100 * correct / total


IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm"]


def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ImageFolderCustomClass(Dataset):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=pil_loader,
        custom_class_to_idx=None,
    ):
        if custom_class_to_idx is None:
            classes, class_to_idx = find_classes(root)
        else:
            class_to_idx = custom_class_to_idx
            classes = list(class_to_idx.keys())
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise (
                RuntimeError(
                    "Found 0 images in subfolders of: " + root + "\n"
                    "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)
                )
            )

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str
