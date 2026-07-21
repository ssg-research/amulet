"""Tests for common/config.py.

The three verification levels are the artifact's cost dial (plan §8): `test` is
tiny-everything, `smoke` runs the real architectures for one epoch, `full` is
paper settings. `full` deliberately leaves `epochs` unset because the paper
epoch count is per-experiment, so the contract under test is: presets are
immutable, `with_defaults` fills only what is unset, and `override` always wins.
"""

import pytest

from common.config import LEVEL_NAMES, LEVELS, LevelConfig, get_level


def test_level_names_are_exactly_the_three_levels() -> None:
    assert set(LEVELS) == {"test", "smoke", "full"}
    assert LEVEL_NAMES == ("test", "smoke", "full")


@pytest.mark.parametrize("name", ["test", "smoke", "full"])
def test_get_level_returns_the_named_preset(name: str) -> None:
    assert get_level(name) == LEVELS[name]
    assert get_level(name).name == name


def test_get_level_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="quick"):
        _ = get_level("quick")


def test_test_level_is_tiny_everything() -> None:
    level = get_level("test")
    assert level.seeds == (0,)
    assert level.epochs == 1
    assert level.train_fraction == 0.01
    assert level.tiny_model is True


def test_smoke_level_uses_real_models_for_one_epoch() -> None:
    level = get_level("smoke")
    assert level.seeds == (0,)
    assert level.epochs == 1
    assert level.train_fraction == 0.1
    assert level.tiny_model is False


def test_full_level_defers_epochs_to_the_experiment() -> None:
    level = get_level("full")
    assert level.seeds == (0,)
    assert level.epochs is None
    assert level.train_fraction == 1.0
    assert level.tiny_model is False


def test_with_defaults_fills_epochs_only_when_unset() -> None:
    assert get_level("full").with_defaults(epochs=90).epochs == 90
    # `test`/`smoke` pin epochs=1 on purpose; a paper setting must not leak in.
    assert get_level("test").with_defaults(epochs=90).epochs == 1


def test_override_wins_over_a_preset_value() -> None:
    level = get_level("smoke").override(seeds=(0, 1, 2), epochs=7, train_fraction=1.0)
    assert level.seeds == (0, 1, 2)
    assert level.epochs == 7
    assert level.train_fraction == 1.0
    assert level.tiny_model is False  # untouched field survives


def test_override_with_no_arguments_is_the_identity() -> None:
    assert get_level("smoke").override() == get_level("smoke")


def test_presets_are_immutable_and_unshared() -> None:
    level = get_level("test")
    with pytest.raises(AttributeError):
        level.epochs = 5  # type: ignore[reportAttributeAccessIssue]
    # Deriving a variant must not mutate the shared preset.
    _ = level.override(epochs=99)
    assert get_level("test").epochs == 1


def test_level_config_is_hashable_so_it_can_key_a_cache() -> None:
    assert isinstance(hash(get_level("test")), int)


def test_levels_are_ordered_by_increasing_cost() -> None:
    fractions = [LEVELS[name].train_fraction for name in LEVEL_NAMES]
    assert fractions == sorted(fractions)


def test_level_config_can_be_constructed_directly() -> None:
    level = LevelConfig(
        name="custom",
        seeds=(3,),
        epochs=2,
        train_fraction=0.5,
        tiny_model=True,
    )
    assert level.seeds == (3,)
