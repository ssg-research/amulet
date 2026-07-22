"""Single source of truth mapping each paper artifact to its `make_*` renderer.

This is the make-side analogue of `common.registry` (which maps experiment IDs
to runner modules): `make/make_all.py` and the test suite both iterate this one
list, so neither can drift from the other, and a paper table or plot cannot be
silently dropped from the regeneration sweep (plan §9).

Each `make_*` module exposes the same two entry points, so a driver treats all
six alike:

* ``generate(results_dir=None, out_dir=None) -> list[Path]`` renders the
  artifact from a results base directory (``results/`` by default, or a
  reviewer's ``runs/<level>/``) into a generated-output directory, returning the
  files written.
* ``coverage_report(results_dir=None) -> list[str]`` reports which cells the
  results directory covers.

The five experiments back six artifacts because E4 feeds both a table and a
figure pair; the figure entry writes two files (a fidelity plot and a
correct-fidelity plot) from one CSV.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from common.paths import artifact_root

if TYPE_CHECKING:
    from types import ModuleType


@dataclass(frozen=True)
class MakeArtifact:
    """One paper artifact and the `make_*` module that regenerates it.

    Attributes:
        artifact_id: Stable ID, the reference file's stem (a table's `.tex` stem
            or, for the figure pair, the shared `fig_outrem` prefix).
        kind: `"table"` (rendered into `tables/generated/`) or `"plot"`
            (rendered into `plots/generated/`).
        experiment_id: The `common.registry` experiment whose CSV backs it.
        module: Dotted path of the `make_*` module exposing `generate` and
            `coverage_report`.
        outputs: The output file stems this artifact writes, without extension
            (the figure entry writes two).
    """

    artifact_id: str
    kind: str
    experiment_id: str
    module: str
    outputs: tuple[str, ...]

    def load(self) -> ModuleType:
        """Import and return the `make_*` module, with `artifact/` on `sys.path`."""
        root = str(artifact_root())
        if root not in sys.path:
            sys.path.insert(0, root)
        return importlib.import_module(self.module)


# The six paper artifacts, in experiment order (E4's table then its figures).
MAKE_ARTIFACTS: tuple[MakeArtifact, ...] = (
    MakeArtifact(
        artifact_id="tab_attack_results",
        kind="table",
        experiment_id="e1_attack_baselines",
        module="make.make_tab_attack_results",
        outputs=("tab_attack_results",),
    ),
    MakeArtifact(
        artifact_id="tab_advtr_modext",
        kind="table",
        experiment_id="e2_advtr_modext",
        module="make.make_tab_advtr_modext",
        outputs=("tab_advtr_modext",),
    ),
    MakeArtifact(
        artifact_id="tab_attinf_advrtr",
        kind="table",
        experiment_id="e3_advtr_attrinf",
        module="make.make_tab_attinf_advrtr",
        outputs=("tab_attinf_advrtr",),
    ),
    MakeArtifact(
        artifact_id="tab_outrem_modext",
        kind="table",
        experiment_id="e4_outrem_modext",
        module="make.make_tab_outrem_modext",
        outputs=("tab_outrem_modext",),
    ),
    MakeArtifact(
        artifact_id="fig_outrem",
        kind="plot",
        experiment_id="e4_outrem_modext",
        module="make.make_fig_outrem",
        outputs=("fig_outrem_fid", "fig_outrem_cor_fid"),
    ),
    MakeArtifact(
        artifact_id="tab_textbadnets_interactions",
        kind="table",
        experiment_id="e5_textbadnets",
        module="make.make_tab_textbadnets",
        outputs=("tab_textbadnets_interactions",),
    ),
)

ARTIFACT_IDS: tuple[str, ...] = tuple(a.artifact_id for a in MAKE_ARTIFACTS)


def get_artifact(artifact_id: str) -> MakeArtifact:
    """Look up a make artifact by its ID.

    Args:
        artifact_id: One of `ARTIFACT_IDS`.

    Returns:
        The matching `MakeArtifact`.

    Raises:
        KeyError: If the ID is not registered.
    """
    for artifact in MAKE_ARTIFACTS:
        if artifact.artifact_id == artifact_id:
            return artifact
    known = ", ".join(ARTIFACT_IDS)
    raise KeyError(f"Unknown artifact {artifact_id!r}. Known IDs: {known}.")
