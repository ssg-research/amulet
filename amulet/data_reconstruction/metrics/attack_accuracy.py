from typing import List
import torch
import numpy as np
from torch.autograd import Variable

def _figure_mse(recover_fig, original_fig):
    diff = torch.nn.MSELoss()
    return diff(recover_fig, original_fig)

def reverse_mse(original_dataset: torch.utils.data.DataLoader,
                reverse_data: List[torch.autograd.Variable],
                input_size: int,
                output_size: int,
                device: str,
) -> float:
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
    class_avg = [Variable(torch.from_numpy(np.zeros(input_size, dtype=np.uint8))).float().to(device) for _ in range(output_size)]
    class_mse = [0 for _ in range(output_size)]
    class_count = [0 for _ in range(output_size)]

    for x, y in original_dataset:
        x, y = x.to(device), y.to(device)
        class_avg[y] = class_avg[y] + x
        class_count[y] = class_count[y] + 1


    for i in range(output_size):
        class_mse[i] = _figure_mse(class_avg[i] / class_count[i], (reverse_data[i]))

    all_class_avg_mse = 0
    for i in range(output_size):
        all_class_avg_mse = all_class_avg_mse + class_mse[i]

    return all_class_avg_mse / output_size
