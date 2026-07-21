"""Single source of truth mapping experiment IDs to their runner modules.

The level sweepers, the `make/` wrappers and the test suite all iterate this one
list, so none of them can drift from the others (plan §9). The five IDs are the
five experiments backing the paper's tables and plots; the two interaction
studies found in the old repository but absent from the paper are deliberately
not here (plan §3).

Modules are named as strings and imported on demand. Importing this module must
stay cheap: resolving all five eagerly would pull in torch, and for E5 the whole
Hugging Face stack, just to answer "which experiments exist?".
"""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING

from .paths import artifact_root

if TYPE_CHECKING:
    from types import ModuleType

# Experiment ID -> runner module, resolved against `artifact/` on `sys.path`.
EXPERIMENTS: dict[str, str] = {
    "e1_attack_baselines": "experiments.e1_attack_baselines.run",
    "e2_advtr_modext": "experiments.e2_advtr_modext.run",
    "e3_advtr_attrinf": "experiments.e3_advtr_attrinf.run",
    "e4_outrem_modext": "experiments.e4_outrem_modext.run",
    "e5_textbadnets": "experiments.e5_textbadnets.run",
}

EXPERIMENT_IDS: tuple[str, ...] = tuple(EXPERIMENTS)


def module_path(experiment_id: str) -> str:
    """Return the import path of an experiment's runner module.

    Args:
        experiment_id: One of `EXPERIMENT_IDS`.

    Returns:
        Dotted module path, e.g. `"experiments.e2_advtr_modext.run"`.

    Raises:
        KeyError: If the ID is not registered.
    """
    if experiment_id not in EXPERIMENTS:
        known = ", ".join(EXPERIMENT_IDS)
        raise KeyError(f"Unknown experiment {experiment_id!r}. Known IDs: {known}.")
    return EXPERIMENTS[experiment_id]


def load_experiment(experiment_id: str) -> ModuleType:
    """Import and return an experiment's runner module.

    Puts the artifact root on `sys.path` first, computed from this file's
    location, so a caller invoked from any working directory resolves
    `experiments.*` without setting `PYTHONPATH`.

    Args:
        experiment_id: One of `EXPERIMENT_IDS`.

    Returns:
        The imported runner module.

    Raises:
        KeyError: If the ID is not registered.
        ModuleNotFoundError: If the runner module has not been built yet.
    """
    path = module_path(experiment_id)
    root = str(artifact_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(path)
