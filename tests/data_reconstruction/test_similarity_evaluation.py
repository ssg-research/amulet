import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from amulet.data_reconstruction.metrics.similarity_evaluation import (
    evaluate_similarity,
)

INPUT_SIZE = (3, 8, 8)
OUTPUT_SIZE = 2


def _make_loader(images: list[torch.Tensor], labels: list[int]) -> DataLoader:
    """Build a batch_size=1 DataLoader over (x, y) pairs as evaluate_similarity expects."""
    x = torch.stack(images)
    y = torch.tensor(labels)
    return DataLoader(TensorDataset(x, y), batch_size=1)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_evaluate_similarity_bounds(seed, assert_nonneg_finite, assert_within):
    # random non-constant images per class, random non-constant reconstructions
    torch.manual_seed(seed)
    images = [torch.rand(*INPUT_SIZE) for _ in range(OUTPUT_SIZE)]
    labels = list(range(OUTPUT_SIZE))
    loader = _make_loader(images, labels)
    reverse_data = [torch.rand(*INPUT_SIZE) for _ in range(OUTPUT_SIZE)]

    results = evaluate_similarity(loader, reverse_data, INPUT_SIZE, OUTPUT_SIZE, "cpu")

    # MSE is a non-negative finite distance, SSIM lies in [-1, 1]
    assert_nonneg_finite(results["mean_mse"])
    assert_within(results["mean_ssim"], -1.0, 1.0)
    for i in range(OUTPUT_SIZE):
        assert_nonneg_finite(results["class_mse"][i])
        assert_within(results["class_ssim"][i], -1.0, 1.0)


def test_perfect_reconstruction_is_zero_mse_unit_ssim():
    # one image per class; reverse_data is the exact same image per class
    torch.manual_seed(0)
    images = [torch.rand(*INPUT_SIZE) for _ in range(OUTPUT_SIZE)]
    labels = list(range(OUTPUT_SIZE))
    loader = _make_loader(images, labels)
    reverse_data = [img.clone() for img in images]

    results = evaluate_similarity(loader, reverse_data, INPUT_SIZE, OUTPUT_SIZE, "cpu")

    # reconstruction matches exactly, so MSE is 0 and SSIM is 1
    assert results["mean_mse"] == pytest.approx(0.0)
    assert results["mean_ssim"] == pytest.approx(1.0)
    for i in range(OUTPUT_SIZE):
        assert results["class_mse"][i] == pytest.approx(0.0)
        assert results["class_ssim"][i] == pytest.approx(1.0)


def test_imperfect_reconstruction_degrades():
    # reconstructions are the class images shifted by a constant offset
    torch.manual_seed(0)
    images = [torch.rand(*INPUT_SIZE) for _ in range(OUTPUT_SIZE)]
    labels = list(range(OUTPUT_SIZE))
    loader = _make_loader(images, labels)
    reverse_data = [img + 0.5 for img in images]

    results = evaluate_similarity(loader, reverse_data, INPUT_SIZE, OUTPUT_SIZE, "cpu")

    # offset reconstruction is worse than a perfect one
    assert results["mean_mse"] > 0
    assert results["mean_ssim"] < 1.0


def test_averages_multiple_images_per_class():
    # two images per class; reconstruction is the exact per-class mean
    torch.manual_seed(0)
    images_class_0 = [torch.rand(*INPUT_SIZE), torch.rand(*INPUT_SIZE)]
    images_class_1 = [torch.rand(*INPUT_SIZE), torch.rand(*INPUT_SIZE)]
    images = images_class_0 + images_class_1
    labels = [0, 0, 1, 1]
    loader = _make_loader(images, labels)
    reverse_data = [
        torch.stack(images_class_0).mean(dim=0),
        torch.stack(images_class_1).mean(dim=0),
    ]

    results = evaluate_similarity(loader, reverse_data, INPUT_SIZE, OUTPUT_SIZE, "cpu")

    # reconstruction equals the per-class average exactly, so MSE is 0
    assert results["mean_mse"] == pytest.approx(0.0)
