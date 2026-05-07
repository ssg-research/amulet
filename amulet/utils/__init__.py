"""
The module amulet.utils contains utilities
for model training and evaluation.
"""

from .__base import (
    get_intermediate_features,
    get_predictions_numpy,
    load_or_train,
    train_classifier,
)
from .__metrics import (
    get_accuracy,
    get_fidelity,
)
from .__pipeline import create_dir, initialize_model, load_data, stratified_split

__all__ = [
    "create_dir",
    "get_accuracy",
    "get_fidelity",
    "get_intermediate_features",
    "get_predictions_numpy",
    "initialize_model",
    "load_data",
    "load_or_train",
    "stratified_split",
    "train_classifier",
]
