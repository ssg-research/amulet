"""Regenerate every paper table and plot from a chosen results directory.

    python artifact/make/make_all.py                      # from committed results/
    python artifact/make/make_all.py --results-dir runs/full   # from a reviewer's re-run

Iterates the make registry (`make.registry.MAKE_ARTIFACTS`) and calls each
`make_*` module's `generate`, writing tables into `tables/generated/` and plots
into `plots/generated/`. Rendering is a pure function of the CSVs: no GPU, no
model, no training, seconds (plan §9, §13, decision 2). An experiment whose CSV
is absent (E1-E4 ship no data yet) renders a correct blank skeleton rather than
crashing, and its per-artifact coverage is reported as MISSING.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from make.registry import MAKE_ARTIFACTS

from common.paths import artifact_root


@dataclass(frozen=True)
class ArtifactResult:
    """The outcome of regenerating one artifact.

    Attributes:
        artifact_id: The artifact's registry ID.
        written: Files the renderer wrote.
        coverage: Per-cell coverage lines from the renderer.
        error: The stringified exception if rendering failed, else None.
    """

    artifact_id: str
    written: list[Path]
    coverage: list[str]
    error: str | None


def generate_all(
    results_dir: Path | None = None,
    tables_dir: Path | None = None,
    plots_dir: Path | None = None,
) -> list[ArtifactResult]:
    """Regenerate every registered artifact from `results_dir`.

    Args:
        results_dir: Base results directory to render from. None uses the
            committed `results/`; a `runs/<level>/` directory renders a
            reviewer's re-run.
        tables_dir: Directory tables are written to. None uses
            `tables/generated/`.
        plots_dir: Directory plots are written to. None uses `plots/generated/`.

    Returns:
        One `ArtifactResult` per registered artifact, in registry order. A
        renderer that raises is captured as an `error` rather than aborting the
        sweep, so one broken artifact does not block the other five.
    """
    tables_dir = (
        artifact_root() / "tables" / "generated" if tables_dir is None else tables_dir
    )
    plots_dir = (
        artifact_root() / "plots" / "generated" if plots_dir is None else plots_dir
    )

    results: list[ArtifactResult] = []
    for artifact in MAKE_ARTIFACTS:
        out_dir = plots_dir if artifact.kind == "plot" else tables_dir
        module = artifact.load()
        try:
            written = module.generate(results_dir=results_dir, out_dir=out_dir)
            coverage = module.coverage_report(results_dir=results_dir)
            results.append(
                ArtifactResult(artifact.artifact_id, written, coverage, None)
            )
        except Exception as exception:
            results.append(
                ArtifactResult(
                    artifact.artifact_id,
                    [],
                    [],
                    f"{type(exception).__name__}: {exception}",
                )
            )
    return results


def _print_report(results: list[ArtifactResult]) -> None:
    """Print per-artifact written paths, coverage, and any error."""
    for result in results:
        print(f"\n=== {result.artifact_id} ===")
        if result.error is not None:
            print(f"  FAILED: {result.error}")
            continue
        for path in result.written:
            print(f"  wrote {path}")
        for line in result.coverage:
            print(line)


def main(argv: list[str] | None = None) -> int:
    """Regenerate every artifact and report per-artifact coverage.

    Returns:
        Process exit code: 0 if every artifact rendered, 1 if any raised.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Base results directory to render from. Default: the committed results/.",
    )
    args = parser.parse_args(argv)

    results = generate_all(results_dir=args.results_dir)
    _print_report(results)

    failed = [result.artifact_id for result in results if result.error is not None]
    if failed:
        print(f"\n{len(failed)} artifact(s) failed to render: {', '.join(failed)}")
        return 1
    print(f"\nregenerated {len(results)} artifact(s) with no GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
