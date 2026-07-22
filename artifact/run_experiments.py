"""Registry-driven runner: run every experiment at one level into runs/<level>/.

    python artifact/run_experiments.py --level test
    python artifact/run_experiments.py --level smoke --only e1,e5
    python artifact/run_experiments.py --level full --seeds 0-4

Iterates `common.registry.EXPERIMENT_IDS` (or a `--only` subset), loads each
experiment's runner via `load_experiment(id)`, and calls its uniform
`run(level=..., seeds=...)`. Every experiment writes under `runs/<level>/` by
default (never the committed `results/`; see `common.io.run_output_dir`), so this
driver passes only the two arguments every runner shares and lets each one place
its own CSVs. A per-experiment pass/fail line and the output location are printed
at the end.

`full` is Level 3: paper settings, GPU-hours. This driver does not guard against
it, but `run_all.sh` prints the warning banner before invoking it.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from common.config import LEVEL_NAMES
from common.io import run_output_dir
from common.registry import EXPERIMENT_IDS, load_experiment


def parse_seeds(text: str) -> tuple[int, ...]:
    """Parse a seed selection such as `0`, `0-4` or `0,2,3`.

    Args:
        text: Comma-separated seeds and inclusive `start-end` ranges.

    Returns:
        The seeds in the order given, without duplicates.

    Raises:
        ValueError: If a part is neither an integer nor an inclusive range.
    """
    seeds: list[int] = []
    for part in text.split(","):
        piece = part.strip()
        if not piece:
            continue
        if "-" in piece.removeprefix("-"):
            start, _, end = piece.partition("-")
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(piece))
    if not seeds:
        raise ValueError(f"No seeds parsed from {text!r}.")
    return tuple(dict.fromkeys(seeds))


def parse_only(text: str) -> tuple[str, ...]:
    """Parse and validate a `--only` subset against the experiment registry.

    Args:
        text: Comma-separated experiment IDs, or `"all"`.

    Returns:
        The requested IDs in registry order, without duplicates.

    Raises:
        ValueError: If any requested ID is not registered.
    """
    if text.strip() == "all":
        return EXPERIMENT_IDS
    requested = [piece.strip() for piece in text.split(",") if piece.strip()]
    unknown = [name for name in requested if name not in EXPERIMENT_IDS]
    if unknown:
        known = ", ".join(EXPERIMENT_IDS)
        raise ValueError(
            f"Unknown experiment(s): {', '.join(unknown)}. Known: {known}."
        )
    return tuple(name for name in EXPERIMENT_IDS if name in set(requested))


@dataclass(frozen=True)
class ExperimentResult:
    """The outcome of running one experiment.

    Attributes:
        experiment_id: The experiment's registry ID.
        ok: Whether `run()` returned without raising.
        rows: Number of result rows appended (0 if all cells were cached).
        seconds: Wall-clock time the run took.
        error: The stringified exception if the run failed, else None.
    """

    experiment_id: str
    ok: bool
    rows: int
    seconds: float
    error: str | None


def run_experiments(
    level: str = "test",
    seeds: tuple[int, ...] | None = None,
    only: tuple[str, ...] = EXPERIMENT_IDS,
    now: float | None = None,
) -> list[ExperimentResult]:
    """Run each requested experiment at `level`, returning per-experiment outcomes.

    Args:
        level: One of `common.config.LEVEL_NAMES`.
        seeds: Seeds to sweep. None keeps each experiment's level-default seeds.
        only: Experiment IDs to run, a subset of `EXPERIMENT_IDS`.
        now: Injectable clock (seconds) for deterministic timing in tests. None
            uses `time.monotonic`.

    Returns:
        One `ExperimentResult` per requested experiment, in registry order. A
        runner that raises is captured as an `error` rather than aborting the
        sweep, so one broken experiment does not block the others.
    """
    clock = (lambda: 0.0) if now is not None else time.monotonic
    results: list[ExperimentResult] = []
    for experiment_id in only:
        module = load_experiment(experiment_id)
        start = clock()
        try:
            rows = module.run(level=level, seeds=seeds)
            results.append(
                ExperimentResult(experiment_id, True, len(rows), clock() - start, None)
            )
        except Exception as exception:
            results.append(
                ExperimentResult(
                    experiment_id,
                    False,
                    0,
                    clock() - start,
                    f"{type(exception).__name__}: {exception}",
                )
            )
    return results


def _print_report(level: str, results: list[ExperimentResult]) -> None:
    """Print a per-experiment pass/fail line and where output landed."""
    output_root = run_output_dir(level)
    print(
        f"\nran {len(results)} experiment(s) at level {level!r}; output under {output_root}"
    )
    for result in results:
        if result.ok:
            print(
                f"  PASS  {result.experiment_id:22} "
                f"{result.rows:>4} row(s) appended in {result.seconds:6.1f}s"
            )
        else:
            print(f"  FAIL  {result.experiment_id:22} {result.error}")


def main(argv: list[str] | None = None) -> int:
    """Run experiments from the command line.

    Returns:
        Process exit code: 0 if every experiment ran, 1 if any raised.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=str, default="test", choices=LEVEL_NAMES)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Seeds to sweep, e.g. `0` or `0-4`. Default: each level's own seeds.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="all",
        help=f"Comma-separated subset of: {', '.join(EXPERIMENT_IDS)}. Default: all.",
    )
    args = parser.parse_args(argv)

    results = run_experiments(
        level=args.level,
        seeds=None if args.seeds is None else parse_seeds(args.seeds),
        only=parse_only(args.only),
    )
    _print_report(args.level, results)

    failed = [result.experiment_id for result in results if not result.ok]
    if failed:
        print(f"\n{len(failed)} experiment(s) failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
