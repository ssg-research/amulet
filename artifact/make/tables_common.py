"""Seed-count-agnostic cell aggregation shared by the interaction-table renderers.

E2 and E3 both render multi-dataset blocks whose cells are means over whatever
seeds the committed CSVs happen to contain: several seeds as
`mean ~$\\pm$~ standard error`, one seed as the bare value, no data as a dash
(plan S7.1, the table contract). That pooling logic is identical for both, so it
lives here rather than in each `make_` script.

Rendering stays a pure function of the CSV rows: no torch, no model, no GPU.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# A cell the table structurally leaves empty, or one with no data yet.
BLANK = "-"


def mean_and_standard_error(values: Sequence[float]) -> tuple[float, float]:
    """Return the mean of `values` and the standard error of that mean.

    Args:
        values: One measurement per seed. At least one.

    Returns:
        `(mean, standard_error)`. A single measurement has no spread, so its
        standard error is zero and the caller renders the value alone.

    Raises:
        ValueError: If `values` is empty.
    """
    if not values:
        raise ValueError("Cannot aggregate an empty cell.")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, 0.0
    return mean, statistics.stdev(values) / len(values) ** 0.5


def format_cell(values: Sequence[float], precision: int = 2) -> str:
    """Render one table cell from the per-seed measurements behind it.

    Args:
        values: One measurement per seed. At least one.
        precision: Decimal places for the mean and the error.

    Returns:
        `"57.17~$\\pm$~0.15"` for several seeds, `"57.17"` for one.

    Raises:
        ValueError: If `values` is empty.
    """
    mean, standard_error = mean_and_standard_error(values)
    if len(values) == 1:
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f}~$\\pm$~{standard_error:.{precision}f}"


def pooled_by_seed(
    rows: Sequence[Mapping[str, str]], column: str, seed_column: str = "exp_id"
) -> list[float]:
    """Collect one measurement per seed for a column, in seed order.

    The seed is the unit of replication, so a seed that contributed several rows
    for a column (which the idempotent append prevents, but a hand-edited CSV
    could hold) is de-duplicated to its last value. Rows whose cell is blank
    (an empty string, as the E3 baseline leaves its robust-accuracy columns) are
    skipped, so a blank never poisons a mean.

    Args:
        rows: The result rows for one cell (one dataset, one budget).
        column: The column to pool.
        seed_column: The column identifying the replication unit.

    Returns:
        One float per seed that carries a value, ordered by seed.
    """
    by_seed: dict[str, float] = {}
    for row in rows:
        raw = row.get(column, "")
        if raw == "" or raw is None:
            continue
        by_seed[row[seed_column]] = float(raw)
    return [by_seed[seed] for seed in sorted(by_seed)]


def cell(rows: Sequence[Mapping[str, str]], column: str, precision: int = 2) -> str:
    """Render a cell from its rows, or a dash when nothing backs it.

    Args:
        rows: The result rows for the cell.
        column: The column the cell renders.
        precision: Decimal places for the value and any error term.

    Returns:
        The formatted `mean ~$\\pm$~ SE` (or bare value), or `BLANK`.
    """
    values = pooled_by_seed(rows, column)
    return format_cell(values, precision) if values else BLANK


def seed_count(
    rows: Sequence[Mapping[str, str]], seed_column: str = "exp_id"
) -> list[str]:
    """Return the distinct seeds present in `rows`, sorted, for a coverage report."""
    return sorted({row[seed_column] for row in rows})


def group_by(
    rows: Sequence[Mapping[str, str]], column: str
) -> dict[str, list[Mapping[str, str]]]:
    """Group rows by the value of one column, preserving row order within a group."""
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[column]].append(row)
    return grouped
