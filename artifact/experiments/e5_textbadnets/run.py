"""Uniform entry point for E5, the text-backdoor experiment (registry `e5_textbadnets`).

Dispatches to the two sub-experiments that share E5's target, dataset and caches:

    python artifact/experiments/e5_textbadnets/run.py --level test
    python artifact/experiments/e5_textbadnets/run.py --level full --seeds 0-4 --which onion

`run(level, seeds, which)` is the same path under a callable name, used by the level
sweepers and the tiny end-to-end test. Both sub-experiments accept their own knobs when
driven directly (`onion.py --help`, `dp.py --help`); this runner exposes only what every
experiment in the artifact exposes, so the sweepers can treat all five alike (plan §9).

Requires the LLM extra: `uv sync --extra cu130 --extra llm` (or `--extra cpu` for
`--level test`, which runs a tiny random-init target on CPU in seconds).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse

from common.config import LEVEL_NAMES, get_level

EXPERIMENT_ID = "e5_textbadnets"

# The paper's epoch count, which the `full` level defers to (plan §7.1).
PAPER_EPOCHS = 3

STUDIES = ("onion", "dp")


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


def run(
    level: str = "full",
    seeds: tuple[int, ...] | None = None,
    which: str = "both",
    output_dir: Path | None = None,
) -> list[dict[str, object]]:
    """Run E5 at one verification level and return the rows it appended.

    Args:
        level: One of `common.config.LEVEL_NAMES`.
        seeds: Seeds to sweep. None keeps the level's own seeds.
        which: `"onion"`, `"dp"` or `"both"`.
        output_dir: Directory the result CSVs go in. None keeps the per-level default
            from `default_output_dir`: `runs/<level>/e5_textbadnets/`, never the
            committed `results/` tree, so no run can overwrite the paper's shipped
            data or have its reduced-budget numbers averaged into them. (The
            throwaway temporary directory `test` uses is its model *cache*, not its
            output; see `_cache_dir`.)

    Returns:
        Every row appended by this call, across the seeds and studies requested. Cells
        already recorded are skipped, so a resumed sweep returns only what it added.

    Raises:
        ValueError: If `which` names no known study.
    """
    if which not in {*STUDIES, "both"}:
        raise ValueError(f"Unknown study {which!r}. Choose onion, dp or both.")
    config = get_level(level).with_defaults(epochs=PAPER_EPOCHS)
    if seeds is not None:
        config = config.override(seeds=tuple(seeds))

    # Imported here, not at module scope, so `--help` and the registry lookup do not pay
    # for torch and the Hugging Face stack.
    from experiments.e5_textbadnets import dp, onion

    modules = {"onion": onion, "dp": dp}
    rows: list[dict[str, object]] = []
    for seed in config.seeds:
        for study in STUDIES:
            if which in (study, "both"):
                rows.extend(modules[study].run_level(config, seed, output_dir))
    return rows


def main(argv: list[str] | None = None) -> None:
    """Run E5 from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=str, default="full", choices=LEVEL_NAMES)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Seeds to sweep, e.g. `0` or `0-4`. Default: the level's own seeds.",
    )
    parser.add_argument("--which", type=str, default="both", choices=[*STUDIES, "both"])
    args = parser.parse_args(argv)
    _ = run(
        level=args.level,
        seeds=None if args.seeds is None else parse_seeds(args.seeds),
        which=args.which,
    )


if __name__ == "__main__":
    main()
