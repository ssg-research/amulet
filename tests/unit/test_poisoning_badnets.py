import torch
from torch.utils.data import TensorDataset

from amulet.poisoning.attacks.badnets import BadNets


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

    # Count how many have the trigger label
    # Original dataset might already have some trigger_label.
    # BadNets poisons 'portion' of the TOTAL dataset from samples NOT having trigger_label.
    # So total poisoned is int(len(dataset) * portion).
    labels = [poisoned_dataset[i][1].item() for i in range(len(poisoned_dataset))]
    original_labels = [tiny_dataset[i][1].item() for i in range(len(tiny_dataset))]

    num_poisoned = sum(
        1
        for i in range(len(labels))
        if labels[i] == trigger_label and original_labels[i] != trigger_label
    )
    assert abs(num_poisoned - int(len(tiny_dataset) * portion)) <= 1


def test_badnets_poison_test_tabular(tiny_dataset):
    trigger_label = 1
    attack = BadNets(
        trigger_label=trigger_label, portion=1.0, random_seed=42, dataset_type="tabular"
    )

    poisoned_dataset = attack.poison_test(tiny_dataset)

    # poison_test only includes samples that were NOT trigger_label
    original_non_trigger_count = sum(
        1 for i in range(len(tiny_dataset)) if tiny_dataset[i][1] != trigger_label
    )
    assert len(poisoned_dataset) == original_non_trigger_count

    for i in range(len(poisoned_dataset)):
        x, y = poisoned_dataset[i]
        assert y == trigger_label
        # Check trigger (tabular: last 1/5 features are 0.0)
        feature_len = x.shape[0]
        assert (x[feature_len - feature_len // 5 :] == 0.0).all()


def test_badnets_poison_image():
    # 3x5x5 image
    x = torch.ones(2, 3, 5, 5)
    y = torch.tensor([0, 1])
    dataset = TensorDataset(x, y)
    attack = BadNets(trigger_label=1, portion=0.5, random_seed=42, dataset_type="image")

    poisoned = attack.poison_train(dataset)

    # Sample 0 was 0, now should be 1 and poisoned.
    x_p, y_p = poisoned[0]
    assert y_p == 1
    # Trigger at [c, w-3, h-3] to [c, w-2, h-2]
    # For 5x5: indices 2, 3.
    assert (x_p[:, 2, 2] == 0.0).all()
    assert (x_p[:, 2, 3] == 0.0).all()
    assert (x_p[:, 3, 2] == 0.0).all()
    assert (x_p[:, 3, 3] == 0.0).all()
    # Others should be 1.0
    assert (x_p[:, 0, 0] == 1.0).all()
