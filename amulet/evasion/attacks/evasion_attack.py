"""Model Evasion Attack Base class
"""
import torch

class EvasionAttack():
    """    
    Base class for evasion
    
    Attributes:
        model: :class:`~torch.nn.Module`
            The model on which to apply adversarial training.
        test_loader: :class:`~torch.utils.data.DataLoader`
            Input data that is perturbed to attack the model.
        device: str
            Device used for model inference. Example: "cuda:0".
        batch_size: int
            Batch size for output data loader.
    """
    def __init__(
            self,
            model: torch.nn.Module,
            test_loader: torch.utils.data.DataLoader,
            device: str,
            batch_size: int
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.batch_size = batch_size
    