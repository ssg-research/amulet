import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from skimage.metrics import structural_similarity


def __figure_mse(recover_fig, original_fig):
    diff = nn.MSELoss()
    return diff(recover_fig, original_fig).item()


def evaluate_similarity(
    original_dataset: DataLoader,
    reverse_data: list[torch.Tensor],
    input_size: tuple[int, ...],
    output_size: int,
    device: str,
) -> dict[str, int | dict[int, float]]:
    """
    Outputs the average MSE loss across different classes

    Args:
        original_dataset: :class:~`torch.utils.data.DataLoader`
            The data used to train the target model, please make sure to set the batch size as 1.
        reverse_data: list of :class:~`torch.autorgrad.Variable`
            The ith element is the reconstructed data point for the ith class.
        input_size: int
            Size of the model's input
        output_size: int
            Size of the model's output
        device: str
            Device on which to load the PyTorch tensors. Example: "cuda:0".
    Returns:
        MSE Loss
    """
    class_total = [
        torch.Tensor(torch.from_numpy(np.zeros(input_size, dtype=np.float32))).to(
            device
        )
        for _ in range(output_size)
    ]
    class_mse = [0 for _ in range(output_size)]
    class_ssim = [0 for _ in range(output_size)]
    class_count = [0 for _ in range(output_size)]

    for x, y in original_dataset:
        x, y = x.to(device), y.to(device)
        class_total[y] = class_total[y] + x
        class_count[y] = class_count[y] + 1

    for i in range(output_size):
        class_avg = class_total[i] / class_count[i]
        class_avg = class_avg.squeeze()
        reverse_data[i] = reverse_data[i].squeeze()
        class_mse[i] = __figure_mse(class_avg, reverse_data[i])
        data_range = reverse_data[i].max().item() - reverse_data[i].min().item()
        class_ssim[i] = structural_similarity(
            class_avg.detach().cpu().numpy(),
            reverse_data[i].detach().cpu().numpy(),
            data_range=data_range,
            channel_axis=0,
        )

    all_class_avg_mse = 0
    all_class_avg_ssim = 0
    per_class_ssim = {}
    per_class_mse = {}
    for i in range(output_size):
        all_class_avg_mse = all_class_avg_mse + class_mse[i]
        all_class_avg_ssim = all_class_avg_ssim + class_ssim[i]
        per_class_mse[i] = class_mse[i]
        per_class_ssim[i] = class_ssim[i]

    mse = all_class_avg_mse / output_size
    ssim = all_class_avg_ssim / output_size

    results = {
        "mean_mse": mse,
        "class_mse": per_class_mse,
        "mean_ssim": ssim,
        "class_ssim": per_class_ssim,
    }

    return results
