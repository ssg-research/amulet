"""Distribution Inference Attack Base class."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ...utils import initialize_model, train_classifier
from ..dataset_utils import DistributionSplits, prepare_distribution_splits


class DistributionInferenceAttack(ABC):
    """
    Base class for distribution inference attacks.

    Distribution inference attacks infer global statistical properties of a
    training distribution, such as the proportion of samples with a given
    sensitive-attribute value, rather than facts about individual records.

    Follows the same lifecycle as MembershipInferenceAttack:
    1. Construct with data and training parameters.
    2. Call prepare_model_populations() to train and cache the four model
       populations (adversary D1, adversary D2, victim D1, victim D2).
    3. Call attack() to run the distinguishing test.

    Attributes:
        x_train: Training features.
        y_train: Training labels, shape (N,).
        z_train: Training sensitive attributes, shape (N, n_sensitive_cols).
        x_test: Test features.
        y_test: Test labels.
        z_test: Test sensitive attributes.
        sensitive_columns: Column names for the z arrays.
        filter_column: Sensitive column whose proportion is being inferred.
        ratio1: Target proportion of filter_column == filter_value for D1.
        ratio2: Target proportion for D2.
        filter_value: Value of filter_column that satisfies the filter.
        drop_values: Per-column values to drop before sampling.
        train_subsample: Minority-class sample count per training draw.
        test_subsample: Minority-class sample count per test draw.
        model_arch: Architecture for population models. Example: "linearnet".
        model_capacity: Capacity for population models. Example: "m1".
        num_features: Number of input features.
        num_classes: Number of output classes.
        num_models: Number of models trained per population.
        epochs: Training epochs per model.
        batch_size: Batch size for DataLoaders and model training.
        device: Device used for training and inference. Example: "cuda:0".
        models_dir: Directory to save and load population model checkpoints.
        dataset: Dataset name used as part of checkpoint filenames.
        exp_id: Experiment ID used as a random seed disambiguator.
    """

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        z_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        z_test: np.ndarray,
        sensitive_columns: list[str],
        filter_column: str,
        ratio1: float,
        ratio2: float,
        model_arch: str,
        model_capacity: str,
        num_features: int,
        num_classes: int,
        num_models: int,
        epochs: int,
        batch_size: int,
        device: str,
        models_dir: Path | str,
        dataset: str,
        exp_id: int = 0,
        filter_value: int = 1,
        drop_values: dict[str, list[int]] | None = None,
        train_subsample: int = 1100,
        test_subsample: int = 500,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.z_train = z_train
        self.x_test = x_test
        self.y_test = y_test
        self.z_test = z_test
        self.sensitive_columns = sensitive_columns
        self.filter_column = filter_column
        self.ratio1 = ratio1
        self.ratio2 = ratio2
        self.filter_value = filter_value
        self.drop_values = drop_values
        self.train_subsample = train_subsample
        self.test_subsample = test_subsample
        self.model_arch = model_arch
        self.model_capacity = model_capacity
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_models = num_models
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.dataset = dataset
        self.exp_id = exp_id

        if isinstance(models_dir, str):
            models_dir = Path(models_dir)
        self.models_dir = models_dir

        self.splits: DistributionSplits | None = None
        self.models_adv_1: list[nn.Module] = []
        self.models_adv_2: list[nn.Module] = []
        self.models_vic_1: list[nn.Module] = []
        self.models_vic_2: list[nn.Module] = []

    def train_model_population(
        self,
        loader: DataLoader,
        *,
        checkpoint_tag: str,
        model_arch: str | None = None,
        model_capacity: str | None = None,
        num_models: int | None = None,
        epochs: int | None = None,
    ) -> list[nn.Module]:
        """
        Train or load a population of models from a single DataLoader.

        Defaults to the instance's stored arch, capacity, num_models, and
        epochs. Pass overrides when a population requires different settings
        (e.g. shadow models with a smaller architecture than the target).

        Args:
            loader: DataLoader supplying training batches.
            checkpoint_tag: Unique string identifying this population within
                models_dir. Convention:
                "{dataset}_dist_inf_{role}_{dist}_{arch}_{capacity}".
                Per-model files: {models_dir}/{checkpoint_tag}_{id}_{exp_id}.pth.
            model_arch: Architecture override. Defaults to self.model_arch.
            model_capacity: Capacity override. Defaults to self.model_capacity.
            num_models: Model count override. Defaults to self.num_models.
            epochs: Epoch count override. Defaults to self.epochs.

        Returns:
            List of trained models set to eval mode with gradients frozen.
        """
        arch = model_arch if model_arch is not None else self.model_arch
        capacity = model_capacity if model_capacity is not None else self.model_capacity
        n_models = num_models if num_models is not None else self.num_models
        n_epochs = epochs if epochs is not None else self.epochs

        self.models_dir.mkdir(parents=True, exist_ok=True)
        criterion = nn.CrossEntropyLoss()
        models: list[nn.Module] = []

        for model_id in range(n_models):
            filename = (
                self.models_dir / f"{checkpoint_tag}_{model_id}_{self.exp_id}.pth"
            )
            model = initialize_model(
                arch, capacity, self.num_features, self.num_classes
            ).to(self.device)

            if filename.exists():
                state = torch.load(
                    filename, weights_only=True, map_location=self.device
                )
                model.load_state_dict(state)
            else:
                print(f"Training {checkpoint_tag} model {model_id + 1}/{n_models}")
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                model = train_classifier(
                    model, loader, criterion, optimizer, n_epochs, self.device
                )
                torch.save(model.state_dict(), filename)

            model.eval()
            model.requires_grad_(False)
            models.append(model)

        return models

    def prepare_model_populations(self) -> None:
        """
        Build distribution splits and train all four model populations.

        Calls prepare_distribution_splits internally, then trains num_models
        models on each of the four resulting loaders (adversary D1, adversary D2,
        victim D1, victim D2). Models are saved to models_dir; already-saved
        checkpoints are loaded rather than retrained.

        Must be called before attack().
        """
        self.splits = prepare_distribution_splits(
            self.x_train,
            self.y_train,
            self.z_train,
            self.x_test,
            self.y_test,
            self.z_test,
            sensitive_columns=self.sensitive_columns,
            filter_column=self.filter_column,
            ratio1=self.ratio1,
            ratio2=self.ratio2,
            train_subsample=self.train_subsample,
            test_subsample=self.test_subsample,
            filter_value=self.filter_value,
            drop_values=self.drop_values,
            batch_size=self.batch_size,
            seed=self.exp_id,
        )

        def _tag(role: str, dist: str) -> str:
            return (
                f"{self.dataset}_dist_inf_{role}_{dist}_"
                f"{self.model_arch}_{self.model_capacity}"
            )

        self.models_adv_1 = self.train_model_population(
            self.splits.adv_trainloader_1, checkpoint_tag=_tag("adv", "1")
        )
        self.models_adv_2 = self.train_model_population(
            self.splits.adv_trainloader_2, checkpoint_tag=_tag("adv", "2")
        )
        self.models_vic_1 = self.train_model_population(
            self.splits.vic_trainloader_1, checkpoint_tag=_tag("vic", "1")
        )
        self.models_vic_2 = self.train_model_population(
            self.splits.vic_trainloader_2, checkpoint_tag=_tag("vic", "2")
        )

    @abstractmethod
    def attack(self) -> dict[str, np.ndarray]:
        pass
