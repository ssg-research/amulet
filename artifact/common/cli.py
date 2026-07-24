"""Command-line parsing helpers shared by every experiment runner.

`parse_seeds` reads the `--seeds` argument every runner accepts (a single seed, a
comma list, or an inclusive `start-end` range), so there is one implementation to
keep correct rather than a copy per experiment. Kept free of torch and of any
experiment import, so a runner's argument parser (and the registry that lists the
experiments) never pulls in the training stack.
"""

from __future__ import annotations


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
