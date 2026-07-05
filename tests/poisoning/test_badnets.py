"""Fast exactness tests for the BadNets backdoor poisoning attack.

BadNets is a mechanical transform: it stamps a fixed trigger and flips labels
to trigger_label, so its output is a deterministic function of the input and
seed. Every assertion here is exact, and the whole file trains nothing.
"""

import pytest
import torch
from torch.utils.data import TensorDataset

from amulet.poisoning.attacks.badnets import BadNets


def _tabular_dataset() -> TensorDataset:
    # 10 features so the tabular trigger (last feature_len // 5 columns) zeros a
    # non-empty slice; with only 4 features it would be 4 // 5 == 0 columns and
    # the stamp would leave x untouched, hiding poisoned points from an
    # x-equality partition.
    torch.manual_seed(42)
    x = torch.rand(64, 10)
    y = torch.randint(0, 2, (64,))
    return TensorDataset(x, y)


def _image_dataset() -> TensorDataset:
    torch.manual_seed(42)
    x = torch.rand(20, 3, 5, 5)
    y = torch.randint(0, 2, (20,))
    return TensorDataset(x, y)


def test_badnets_poison_train_tabular(tiny_dataset):
    trigger_label = 1
    portion = 0.5
    attack = BadNets(
        trigger_label=trigger_label,
        portion=portion,
        random_seed=42,
        dataset_type="tabular",
    )

    poisoned_dataset = attack.poison_train(tiny_dataset)

    assert len(poisoned_dataset) == len(tiny_dataset)

    # BadNets draws its poison set only from points NOT already carrying
    # trigger_label, so the count is int(N*portion) capped by how many such
    # points exist — here 31 non-trigger points cap the 32-sample target.
    labels = [poisoned_dataset[i][1].item() for i in range(len(poisoned_dataset))]
    original_labels = [tiny_dataset[i][1].item() for i in range(len(tiny_dataset))]

    num_non_trigger = sum(1 for lbl in original_labels if lbl != trigger_label)
    num_poisoned = sum(
        1
        for i in range(len(labels))
        if labels[i] == trigger_label and original_labels[i] != trigger_label
    )
    assert num_poisoned == min(int(len(tiny_dataset) * portion), num_non_trigger)


@pytest.mark.parametrize(
    "dataset_type, make_dataset",
    [("tabular", _tabular_dataset), ("image", _image_dataset)],
)
def test_badnets_poison_train_clean_points_byte_identical(dataset_type, make_dataset):
    """Points BadNets does not poison must be returned unchanged, both x and y.

    A point is stamped iff its features changed; the source data uses random
    floats so a genuine trigger stamp (zeros) can never coincide with the
    original values. So x-equality cleanly separates clean from poisoned, and
    each partition gets its exact invariant.
    """
    dataset = make_dataset()
    attack = BadNets(
        trigger_label=1, portion=0.5, random_seed=42, dataset_type=dataset_type
    )

    poisoned = attack.poison_train(dataset)

    for i in range(len(dataset)):
        orig_x, orig_y = dataset[i]
        out_x, out_y = poisoned[i]
        if torch.equal(out_x, orig_x):
            assert out_y.item() == int(orig_y.item())
        else:
            assert out_y.item() == 1
            assert int(orig_y.item()) != 1


def test_badnets_poison_test_tabular(tiny_dataset):
    trigger_label = 1
    attack = BadNets(
        trigger_label=trigger_label, portion=1.0, random_seed=42, dataset_type="tabular"
    )

    poisoned_dataset = attack.poison_test(tiny_dataset)

    # poison_test returns exactly the samples that were NOT already trigger_label.
    original_non_trigger_count = sum(
        1 for i in range(len(tiny_dataset)) if tiny_dataset[i][1] != trigger_label
    )
    assert len(poisoned_dataset) == original_non_trigger_count

    for i in range(len(poisoned_dataset)):
        x, y = poisoned_dataset[i]
        # Every returned point is relabeled to trigger_label ...
        assert y == trigger_label
        # ... and carries the tabular trigger (last 1/5 features zeroed).
        feature_len = x.shape[0]
        assert (x[feature_len - feature_len // 5 :] == 0.0).all()


def test_badnets_poison_image():
    x = torch.ones(2, 3, 5, 5)
    y = torch.tensor([0, 1])
    dataset = TensorDataset(x, y)
    attack = BadNets(trigger_label=1, portion=0.5, random_seed=42, dataset_type="image")

    poisoned = attack.poison_train(dataset)

    x_p, y_p = poisoned[0]
    assert y_p == 1
    # Trigger is a 2x2 block at rows/cols {w-3, w-2} = {2, 3} for a 5x5 image.
    assert (x_p[:, 2, 2] == 0.0).all()
    assert (x_p[:, 2, 3] == 0.0).all()
    assert (x_p[:, 3, 2] == 0.0).all()
    assert (x_p[:, 3, 3] == 0.0).all()
    # Pixels outside the trigger keep their original value.
    assert (x_p[:, 0, 0] == 1.0).all()


def test_badnets_invalid_dataset_type_raises(tiny_dataset):
    # The type check lives in the per-point stamp, so it fires the moment a
    # point is actually poisoned; poison_test stamps every non-trigger point.
    attack = BadNets(trigger_label=1, portion=1.0, random_seed=42, dataset_type="bogus")
    with pytest.raises(ValueError):
        attack.poison_test(tiny_dataset)


def test_badnets_same_seed_identical_poison(tiny_dataset):
    """Same random_seed selects the same poisoned indices, so the two poisoned
    datasets are byte-identical throughout (a superset of 'same indices')."""
    run_a = BadNets(
        trigger_label=1, portion=0.5, random_seed=42, dataset_type="tabular"
    ).poison_train(tiny_dataset)
    run_b = BadNets(
        trigger_label=1, portion=0.5, random_seed=42, dataset_type="tabular"
    ).poison_train(tiny_dataset)

    for i in range(len(tiny_dataset)):
        assert torch.equal(run_a[i][0], run_b[i][0])
        assert run_a[i][1].item() == run_b[i][1].item()
