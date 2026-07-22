"""Contract for the make registry and `make_all.py` (plan §9, P5).

The make registry is the single source of truth both `make_all.py` and these
tests iterate, so a paper artifact cannot silently fall out of the regeneration
sweep. These tests pin: the registry lists exactly the six paper artifacts, each
maps to a real `make_*` module with the uniform `generate` / `coverage_report`
interface, and `make_all` regenerates all six from the shipped `results/` with no
GPU, rendering a blank skeleton (not a crash) for the four experiments that ship
no CSV yet and reproducing E5's committed table exactly.

Pure rendering: no torch, no model, no training. The `make_*` modules import
matplotlib (for the figure) and the experiment schemas, never torch at module
scope, so this whole file runs on the fast tier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from make import make_all
from make.registry import ARTIFACT_IDS, MAKE_ARTIFACTS, get_artifact

from common.paths import artifact_root
from common.registry import EXPERIMENT_IDS

if TYPE_CHECKING:
    from pathlib import Path

# The six paper artifacts P5 must always regenerate: five tables and one figure
# pair. Named here so a drift in the registry (an added or dropped artifact) is
# a red test, not a silently shorter sweep.
_EXPECTED_ARTIFACTS = {
    "tab_attack_results",
    "tab_advtr_modext",
    "tab_attinf_advrtr",
    "tab_outrem_modext",
    "fig_outrem",
    "tab_textbadnets_interactions",
}


def test_registry_lists_exactly_the_six_paper_artifacts() -> None:
    """The registry holds the six paper artifacts, no more, no fewer, no dupes."""
    assert set(ARTIFACT_IDS) == _EXPECTED_ARTIFACTS
    assert len(MAKE_ARTIFACTS) == len(_EXPECTED_ARTIFACTS)
    assert len(ARTIFACT_IDS) == len(set(ARTIFACT_IDS))


def test_every_artifact_maps_to_a_real_make_module() -> None:
    """Each registry entry imports and exposes the uniform generate/coverage API."""
    for artifact in MAKE_ARTIFACTS:
        module = artifact.load()
        assert callable(getattr(module, "generate", None)), artifact.module
        assert callable(getattr(module, "coverage_report", None)), artifact.module


def test_every_artifact_is_backed_by_a_registered_experiment() -> None:
    """No artifact points at an experiment the runner registry does not know."""
    for artifact in MAKE_ARTIFACTS:
        assert artifact.experiment_id in EXPERIMENT_IDS, artifact.artifact_id


def test_every_experiment_backs_at_least_one_artifact() -> None:
    """All five experiments feed the paper; none is dropped from regeneration."""
    covered = {artifact.experiment_id for artifact in MAKE_ARTIFACTS}
    assert covered == set(EXPERIMENT_IDS)


def test_each_artifact_kind_is_table_or_plot() -> None:
    """Kind routes an artifact to tables/generated or plots/generated."""
    for artifact in MAKE_ARTIFACTS:
        assert artifact.kind in ("table", "plot"), artifact.artifact_id


def test_get_artifact_round_trips_and_rejects_unknown() -> None:
    """`get_artifact` finds a registered ID and raises on an unknown one."""
    import pytest

    assert get_artifact("fig_outrem").kind == "plot"
    with pytest.raises(KeyError):
        _ = get_artifact("tab_does_not_exist")


def test_make_all_regenerates_every_artifact_without_gpu(tmp_path: Path) -> None:
    """`make_all` renders all six from shipped results/, none failing, all writing.

    No experiment CSV is required for a table to render: the four experiments
    that ship no data (E1-E4) render a blank skeleton, so the sweep completes
    with zero errors and every artifact writes at least one output file.
    """
    results = make_all.generate_all(
        tables_dir=tmp_path / "tables", plots_dir=tmp_path / "plots"
    )

    assert {result.artifact_id for result in results} == _EXPECTED_ARTIFACTS
    assert all(result.error is None for result in results), [
        (r.artifact_id, r.error) for r in results if r.error
    ]
    assert all(result.written for result in results)


def test_make_all_reproduces_the_committed_e5_table(tmp_path: Path) -> None:
    """From shipped results/, the E5 table regenerates byte-for-byte (plan §13).

    E5 is the one experiment whose ground-truth CSV ships, so its table is the
    reproducibility proof: `make_all` rebuilds it identically to the committed
    reference, GPU-free.
    """
    _ = make_all.generate_all(
        tables_dir=tmp_path / "tables", plots_dir=tmp_path / "plots"
    )

    reference = (
        artifact_root() / "tables" / "tab_textbadnets_interactions.tex"
    ).read_text()
    generated = (tmp_path / "tables" / "tab_textbadnets_interactions.tex").read_text()
    assert generated == reference


def test_make_all_renders_a_blank_skeleton_for_an_experiment_with_no_data(
    tmp_path: Path,
) -> None:
    """An experiment shipping no CSV renders a valid, data-free table skeleton."""
    _ = make_all.generate_all(
        tables_dir=tmp_path / "tables", plots_dir=tmp_path / "plots"
    )

    e2_table = (tmp_path / "tables" / "tab_advtr_modext.tex").read_text()
    assert e2_table.startswith("\\begin{table")
    assert e2_table.rstrip().endswith("\\end{table*}")
    assert "$\\pm$" not in e2_table  # no aggregated cells: nothing was measured


def test_make_all_reads_a_runs_directory_when_asked(tmp_path: Path) -> None:
    """`--results-dir` points the sweep at a reviewer's runs/<level>/ tree.

    A results base with no CSVs at all still renders every skeleton without
    crashing, proving the base-dir plumbing reaches each renderer.
    """
    empty_base = tmp_path / "runs" / "full"
    empty_base.mkdir(parents=True)

    results = make_all.generate_all(
        results_dir=empty_base,
        tables_dir=tmp_path / "tables",
        plots_dir=tmp_path / "plots",
    )

    assert all(result.error is None for result in results)
    assert len(results) == len(_EXPECTED_ARTIFACTS)
