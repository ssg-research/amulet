"""Every dataset an experiment sweeps must be one the library can load.

The experiments carry their own dataset labels: `cifar` is what the paper's
tables and figures label the column, what the result CSVs record, and what the
model-spec cache keys are built from. `amulet.utils.load_data` knows the same
dataset as `cifar10` and raises `ValueError: Unknown dataset` for anything else,
so the two vocabularies are bridged by `shared_targets.loader_name`.

That bridge is invisible at `--level test`, which substitutes synthetic tabular
data for every dataset and never calls `load_data` at all. A label that no
loader answers to therefore passes the whole fast tier and fails hours into a
smoke or full run. This test closes that gap without a download, by checking the
labels against the names `load_data` actually dispatches on.
"""

from __future__ import annotations

import inspect

import pytest

from experiments.e2_advtr_modext.schemas import DATASETS as E2_DATASETS
from experiments.e3_advtr_attrinf.schemas import DATASETS as E3_DATASETS
from experiments.e4_outrem_modext.schemas import DATASETS as E4_DATASETS
from experiments.shared_targets import loader_name


def library_dataset_names() -> frozenset[str]:
    """Return the dataset names `amulet.utils.load_data` dispatches on.

    Read out of the function's own source rather than duplicated here, so a
    loader added or renamed in the library is reflected without editing a list
    in the test.

    Returns:
        Every literal `name` in a `dataset == "name"` branch of `load_data`.
    """
    from amulet.utils import load_data

    names: set[str] = set()
    for line in inspect.getsource(load_data).splitlines():
        marker = 'dataset == "'
        if marker not in line:
            continue
        rest = line.split(marker, 1)[1]
        names.add(rest.split('"', 1)[0])
    return frozenset(names)


def test_library_dataset_names_were_found() -> None:
    """The source scrape found branches, so an empty result cannot pass vacuously."""
    names = library_dataset_names()
    assert "celeba" in names
    assert "cifar10" in names


@pytest.mark.parametrize("dataset", sorted({*E2_DATASETS, *E3_DATASETS, *E4_DATASETS}))
def test_every_swept_dataset_resolves_to_a_library_loader(dataset: str) -> None:
    """Each experiment label maps to a name `load_data` accepts."""
    resolved = loader_name(dataset)
    assert resolved in library_dataset_names(), (
        f"Experiments sweep {dataset!r}, which resolves to {resolved!r}, but "
        f"load_data only knows {sorted(library_dataset_names())}. Add an entry "
        f"to shared_targets.LOADER_NAMES."
    )


def test_labels_without_an_override_pass_through_unchanged() -> None:
    """`loader_name` only rewrites the labels that genuinely differ."""
    assert loader_name("census") == "census"
    assert loader_name("cifar") == "cifar10"
