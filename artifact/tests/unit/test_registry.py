"""Tests for common/registry.py.

The registry is the single source of truth mapping experiment IDs to modules, so
the sweepers, the `make/` wrappers and the tests all iterate one list and cannot
drift (plan §9). Resolution is lazy on purpose: importing the registry must not
import five experiment modules (and their torch/transformers stacks).
"""

import pytest

from common.registry import EXPERIMENT_IDS, EXPERIMENTS, module_path

EXPECTED_IDS = (
    "e1_attack_baselines",
    "e2_advtr_modext",
    "e3_advtr_attrinf",
    "e4_outrem_modext",
    "e5_textbadnets",
)


def test_the_registry_holds_exactly_the_five_paper_experiments() -> None:
    assert EXPERIMENT_IDS == EXPECTED_IDS
    assert set(EXPERIMENTS) == set(EXPECTED_IDS)


@pytest.mark.parametrize("experiment_id", EXPECTED_IDS)
def test_each_id_maps_to_its_own_runner_module(experiment_id: str) -> None:
    assert module_path(experiment_id) == f"experiments.{experiment_id}.run"


def test_module_paths_are_unique() -> None:
    paths = [module_path(experiment_id) for experiment_id in EXPERIMENT_IDS]
    assert len(set(paths)) == len(paths)


def test_module_path_rejects_an_unknown_id() -> None:
    with pytest.raises(KeyError, match="e6_mystery"):
        _ = module_path("e6_mystery")


def test_importing_the_registry_does_not_import_the_experiments() -> None:
    import sys

    assert not [name for name in sys.modules if name.startswith("experiments.")]
