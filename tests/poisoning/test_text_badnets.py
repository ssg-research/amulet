"""Exactness tests for the TextBadNets textual backdoor poisoning attack.

Like the image BadNets, TextBadNets is a mechanical, deterministic transform: it inserts
a trigger phrase in string space and flips labels to trigger_label. The string-space
``poison_text`` needs no tokenizer and is asserted exactly here. The ``poison_train`` /
``poison_test`` tests additionally tokenize, so they use the ``tiny_text_dataset`` fixture
(which skips when the tokenizer is unavailable offline).
"""

import pytest
import torch

from amulet.datasets import TextTensorDataset
from amulet.poisoning.attacks import TextBadNets


def _attack(insert_position: str = "start", **kwargs) -> TextBadNets:
    params = {
        "trigger": "cf",
        "trigger_label": 1,
        "portion": 0.5,
        "random_seed": 42,
        "insert_position": insert_position,
    }
    params.update(kwargs)
    return TextBadNets(**params)  # type: ignore[arg-type]


# --- Pure string-space transform: no tokenizer, always runs ---------------------------


@pytest.mark.parametrize(
    "position, expected",
    [("start", "cf a good movie"), ("end", "a good movie cf")],
)
def test_poison_text_inserts_trigger_at_position(position, expected):
    assert _attack(position).poison_text("a good movie") == expected


def test_poison_text_random_inserts_trigger_deterministically():
    attack = _attack("random", trigger="XX")
    sentence = "the movie was really quite good indeed"
    first = attack.poison_text(sentence)
    second = attack.poison_text(sentence)
    # Pure and reproducible: same input + seed -> same output.
    assert first == second
    # The trigger is inserted as its own word, not at an end.
    words = first.split()
    assert "XX" in words
    assert words[0] != "XX" and words[-1] != "XX"
    # Every original word is preserved, in order.
    assert [w for w in words if w != "XX"] == sentence.split()


def test_invalid_insert_position_raises():
    with pytest.raises(ValueError):
        _attack("middle")


# --- Poison-index selection: pure, no tokenizer, runs in the fast tier ----------------
# poison_train/test re-tokenize (needing the llm extra), but the row-selection logic is a
# pure function of labels + seed, so it is guarded here without a tokenizer.

_LABELS = [1, 0, 1, 0, 0, 1, 0, 0]  # 3 target (label 1), 5 non-target (label 0)


def test_select_poison_indices_only_picks_non_target_rows():
    selected = _attack(portion=0.5).select_poison_indices(_LABELS, len(_LABELS))
    assert all(_LABELS[i] != 1 for i in selected)


@pytest.mark.parametrize(
    "portion, expected",
    [
        (0.0, 0),  # nothing requested
        (0.5, 4),  # int(8 * 0.5) = 4, and 5 non-target rows exist
        (1.0, 5),  # int(8 * 1.0) = 8, capped by the 5 non-target rows
    ],
)
def test_select_poison_indices_count_capped_by_non_target(portion, expected):
    selected = _attack(portion=portion).select_poison_indices(_LABELS, len(_LABELS))
    assert len(selected) == expected


def test_select_poison_indices_is_deterministic():
    a = _attack(portion=0.5).select_poison_indices(_LABELS, len(_LABELS))
    b = _attack(portion=0.5).select_poison_indices(_LABELS, len(_LABELS))
    assert a == b


# --- poison_train / poison_test: need the tokenizer (skips offline if unavailable) ----


def test_poison_train_relabels_and_triggers_selected_rows(tiny_text_dataset):
    attack = _attack("start", portion=0.5)
    poisoned = attack.poison_train(tiny_text_dataset)

    assert isinstance(poisoned, TextTensorDataset)
    assert len(poisoned) == len(tiny_text_dataset)

    # trigger_label=1, portion=0.5 over 4 rows -> target 2, and exactly the 2 non-target
    # (label 0) rows exist, so both are poisoned.
    triggered = [i for i, t in enumerate(poisoned.texts) if t.startswith("cf ")]
    assert len(triggered) == 2
    for i in triggered:
        assert poisoned[i][1].item() == 1
        assert int(tiny_text_dataset[i][1].item()) == 0


def test_poison_train_leaves_clean_rows_unchanged(tiny_text_dataset):
    attack = _attack("start", portion=0.5)
    poisoned = attack.poison_train(tiny_text_dataset)

    for i in range(len(tiny_text_dataset)):
        if not poisoned.texts[i].startswith("cf "):
            assert poisoned.texts[i] == tiny_text_dataset.texts[i]
            assert poisoned[i][1].item() == int(tiny_text_dataset[i][1].item())
            assert torch.equal(poisoned[i][0], tiny_text_dataset[i][0])


def test_poison_train_is_reproducible(tiny_text_dataset):
    run_a = _attack("start", portion=0.5).poison_train(tiny_text_dataset)
    run_b = _attack("start", portion=0.5).poison_train(tiny_text_dataset)

    assert run_a.texts == run_b.texts
    assert torch.equal(run_a.tensors[0], run_b.tensors[0])
    assert torch.equal(run_a.tensors[1], run_b.tensors[1])


def test_poison_train_returns_padded_text_tensor_dataset(tiny_text_dataset):
    attack = _attack("start", portion=0.5)
    poisoned = attack.poison_train(tiny_text_dataset)

    max_len = tiny_text_dataset.tensors[0].shape[1]
    assert poisoned.tensors[0].shape == (len(tiny_text_dataset), max_len)
    assert poisoned.tensors[0].dtype == torch.int64
    assert poisoned.tokenizer_name == tiny_text_dataset.tokenizer_name


def test_poison_test_triggers_every_non_target_row(tiny_text_dataset):
    attack = _attack("start", portion=1.0)
    poisoned = attack.poison_test(tiny_text_dataset)

    num_non_target = sum(
        1
        for i in range(len(tiny_text_dataset))
        if int(tiny_text_dataset[i][1].item()) != 1
    )
    assert len(poisoned) == num_non_target
    assert all(t.startswith("cf ") for t in poisoned.texts)
    assert all(poisoned[i][1].item() == 1 for i in range(len(poisoned)))
