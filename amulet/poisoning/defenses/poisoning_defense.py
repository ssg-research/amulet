"""Base class for poisoning defenses"""
from typing import Optional, Callable
import torch
from ...utils import train_classifier

class PoisoningDefense():
    """    
    Base class for Poisoning defenses.

   
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
        device: str
            Device used to train model. Example: "cuda:0".
        epochs: int
            Determines number of iterations over training data.
        batch_size: int 
            Batch size of training data.
    """
    def __init__(
            self,
            model: torch.nn.Module,
            criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader,
            device: str,
            train_function: Optional[Callable[[torch.nn.Module, 
                                               torch.utils.data.DataLoader, 
                                               torch.nn.Module, 
                                               torch.optim.Optimizer, 
                                               int, 
                                               str], 
                                               torch.nn.Module]] = train_classifier,
            epochs: Optional[int] = 50,
            batch_size: Optional[int] = 256
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.train = train_function
        self.epochs = epochs
        self.batch_size = batch_size
