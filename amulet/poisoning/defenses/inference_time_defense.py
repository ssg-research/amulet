"""Base class for inference-time input-purification poisoning defenses."""

from abc import ABC, abstractmethod

from torch.utils.data import TensorDataset


class InferenceTimeDefense(ABC):
    """Base class for poisoning defenses that purify inputs at inference time.

    ``PoisoningDefense`` assumes model retraining (it exposes ``train_robust`` and
    carries a model, optimizer, and loaders). An input-purification defense such as
    ONION does not retrain: it takes a dataset, removes likely-trigger content, and
    returns a cleaned dataset that the (already trained) victim then classifies. This
    base captures that different contract, mirroring how watermarking and
    fingerprinting keep their own ABCs rather than contorting a poor-fitting one.

    Like every Amulet attack and defense, ``purify`` emits no metric; it returns an
    artifact (the purified dataset). Metrics are computed separately.
    """

    @abstractmethod
    def purify(self, dataset: TensorDataset) -> TensorDataset:
        """Return a purified copy of ``dataset`` with likely-trigger content removed."""
