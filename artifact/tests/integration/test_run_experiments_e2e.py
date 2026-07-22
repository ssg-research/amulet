"""End-to-end wiring of the P5 driver path at `test` level (plan §8 L1, §9, P5).

This exercises the same two-step path `run_smoke.sh` runs — drive every
experiment through `run_experiments`, then regenerate the artifacts from the run
directory with `make_all` — but at `test` level (tiny synthetic data, CPU,
seconds) rather than `smoke` (real models, a GPU, dataset downloads). It runs one
experiment (E4, which needs no `llm` extra) and asserts the two load-bearing P5
properties: the run's output lands under `runs/<level>/`, and the committed
`results/` tree is left byte-for-byte untouched.

`run_smoke.sh` itself hard-wires `--level smoke`; running it for real needs a GPU
and is out of scope for the fast tier, so its shell wiring is dry-verified
(`bash -n`, and the checkpoint) rather than executed here.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _tree_digest(root: Path) -> dict[str, str]:
    """Map each file under `root` to a content hash, for an untouched-check."""
    if not root.exists():
        return {}
    digests: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digests[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return digests


@pytest.mark.integration
def test_driver_writes_under_runs_and_leaves_results_untouched() -> None:
    """`run_experiments` at test level writes E4's CSV under runs/, not results/."""
    import run_experiments

    from common.io import results_root, run_output_dir

    committed_before = _tree_digest(results_root())

    results = run_experiments.run_experiments(
        level="test", seeds=(0,), only=("e4_outrem_modext",)
    )

    assert len(results) == 1
    assert results[0].ok, results[0].error

    run_csv = run_output_dir("test") / "e4_outrem_modext.csv"
    assert run_csv.exists(), f"E4 did not write under runs/test: {run_csv}"

    # The committed ground truth is untouched, and no E4 CSV appeared in results/.
    assert _tree_digest(results_root()) == committed_before
    assert not (results_root() / "e4_outrem_modext.csv").exists()


@pytest.mark.integration
def test_make_all_renders_the_e4_table_from_the_run_directory(tmp_path: Path) -> None:
    """The run's CSV feeds `make_all` with `--results-dir runs/test`, non-blank.

    This closes the loop `run_smoke.sh` runs: experiments write under runs/, then
    `make_all` regenerates the artifacts from that same directory. Rendering E4
    from the run directory must produce a table backed by the tiny run's data
    (coverage is not MISSING), proving the base-dir plumbing carries end to end.
    """
    import run_experiments
    from make import make_all

    from common.io import run_output_dir

    _ = run_experiments.run_experiments(
        level="test", seeds=(0,), only=("e4_outrem_modext",)
    )
    run_dir = run_output_dir("test")

    results = make_all.generate_all(
        results_dir=run_dir,
        tables_dir=tmp_path / "tables",
        plots_dir=tmp_path / "plots",
    )

    e4 = next(r for r in results if r.artifact_id == "tab_outrem_modext")
    assert e4.error is None
    assert any("MISSING" not in line for line in e4.coverage), e4.coverage
    assert (tmp_path / "tables" / "tab_outrem_modext.tex").read_text().count("\\\\") > 0
