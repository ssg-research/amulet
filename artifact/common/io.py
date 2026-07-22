"""Result-CSV layout, schema validation and idempotent appends.

Every experiment writes one row per configuration into `artifact/results/`, and
those CSVs are committed so each `make/` wrapper can render its table or plot in
seconds with no GPU (plan §13, decision 2). Two properties follow from that:

* **Appends are idempotent.** A sweep that dies halfway is resumed by re-running
  it; rows whose key columns already appear are skipped rather than duplicated.
  Key columns identify an experiment *cell* (dataset, seed, epsilon, ...), not
  the measured values, which may differ in the last decimal on a recompute.
* **The header is validated.** A schema that has gained a column since the CSV
  was written is an error, not a silently misaligned row.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .paths import artifact_root

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


@dataclass(frozen=True)
class CsvSchema:
    """Column layout of one experiment's result CSV.

    Attributes:
        header: Every column, in the order they are written.
        key_columns: The subset of `header` identifying an experiment cell. Two
            rows agreeing on all of these are the same measurement repeated.
    """

    header: tuple[str, ...]
    key_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject key columns that are not part of the header.

        Raises:
            ValueError: If any key column is absent from `header`.
        """
        unknown = [column for column in self.key_columns if column not in self.header]
        if unknown:
            raise ValueError(
                f"Key columns absent from the header: {', '.join(unknown)}. "
                f"Header is: {', '.join(self.header)}."
            )

    def validate_row(self, row: Mapping[str, object]) -> None:
        """Check that a row supplies exactly the schema's columns.

        Args:
            row: Column name to value.

        Raises:
            ValueError: If the row is missing a column or carries an unknown one.
        """
        missing = [column for column in self.header if column not in row]
        unknown = [column for column in row if column not in self.header]
        if missing or unknown:
            problems: list[str] = []
            if missing:
                problems.append(f"missing columns: {', '.join(missing)}")
            if unknown:
                problems.append(f"unknown columns: {', '.join(unknown)}")
            raise ValueError(f"Row does not match the schema ({'; '.join(problems)}).")


def results_root() -> Path:
    """Return the directory holding the committed, ground-truth result CSVs.

    This tree is git-tracked and read-only to the harness: no experiment's
    `run()` writes here by default (all runs go to `runs/<level>/`, see
    `run_output_dir`). Authors promote a completed L3 run by copying
    `runs/full/<id>.csv` into here and committing.

    Returns:
        `artifact/results`, which is git-tracked.
    """
    return artifact_root() / "results"


def runs_root() -> Path:
    """Return the directory holding harness run outputs, one subtree per level.

    Every `run()` (test, smoke or full) writes under `runs/<level>/` by default,
    never into the committed `results/` tree. This directory is gitignored: it
    holds reviewer re-runs and author sweeps, not shipped ground truth.

    Returns:
        `artifact/runs`, which is gitignored.
    """
    return artifact_root() / "runs"


def run_output_dir(level_name: str) -> Path:
    """Return the results directory a run at `level_name` writes under.

    The `runs/<level>/` tree mirrors the `results/` layout exactly (E1 and E5
    write into a `<experiment_id>/` subdirectory of it; E2/E3/E4 write
    `<experiment_id>.csv` directly into it), so a `make_*` renderer can read a
    reviewer's `runs/full/` with the same path logic it uses for `results/`.

    Args:
        level_name: The level, one of `common.config.LEVEL_NAMES`.

    Returns:
        `artifact/runs/<level_name>`, created by the caller on first write.
    """
    return runs_root() / level_name


def results_path(
    experiment_id: str, stem: str | None = None, base: Path | None = None
) -> Path:
    """Return where an experiment's result CSV lives, under `base`.

    An experiment producing a single CSV gets `<base>/<experiment_id>.csv`. One
    producing several (E5 writes an ONION CSV and a DP CSV) passes a `stem` and
    gets `<base>/<experiment_id>/<stem>.csv`, keeping them together.

    Args:
        experiment_id: Registry ID, e.g. `"e2_advtr_modext"`.
        stem: Filename stem for experiments writing more than one CSV.
        base: Directory the layout is rooted at. Defaults to `results_root()`
            (the shipped ground truth); pass a `runs/<level>/` directory to
            resolve a reviewer's re-run in the same layout.

    Returns:
        Path to the CSV. The file may not exist yet.
    """
    root = results_root() if base is None else base
    if stem is None:
        return root / f"{experiment_id}.csv"
    return root / experiment_id / f"{stem}.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a result CSV into a list of string-valued rows.

    Args:
        path: Path to the CSV.

    Returns:
        One dict per data row, keyed by column name. Empty if the file does not
        exist, so a caller need not special-case a sweep that has not run yet.
    """
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _matches(cell: str, value: object) -> bool:
    """Compare a CSV cell against the value a caller wants to write.

    Numbers compare by value, because `0.10`, `0.1` and `1e-1` all name the same
    experiment cell but render differently. Everything else compares as text.
    """
    if isinstance(value, bool):
        return cell == str(value)
    if isinstance(value, int | float):
        try:
            return float(cell) == float(value)
        except ValueError:
            return False
    return cell == str(value)


def row_exists(path: Path, schema: CsvSchema, row: Mapping[str, object]) -> bool:
    """Report whether a row with these key-column values is already recorded.

    Args:
        path: Path to the CSV.
        schema: Schema whose `key_columns` identify the cell.
        row: The row whose key columns are looked up.

    Returns:
        True if an existing row agrees on every key column.
    """
    return any(
        all(_matches(existing[column], row[column]) for column in schema.key_columns)
        for existing in read_rows(path)
    )


def append_row(path: Path, schema: CsvSchema, row: Mapping[str, object]) -> bool:
    """Append a row to a result CSV unless its cell is already recorded.

    Creates the file (and any missing parent directories) with a header row on
    first write. Re-running a completed sweep is therefore a no-op.

    Args:
        path: Path to the CSV.
        schema: Schema defining the header and the cell-identifying columns.
        row: Column name to value, matching the schema exactly.

    Returns:
        True if the row was written, False if it was already present.

    Raises:
        ValueError: If the row does not match the schema, or the existing file's
            header does not.
    """
    schema.validate_row(row)

    if path.exists():
        with path.open(newline="") as handle:
            existing_header = next(csv.reader(handle), [])
        if tuple(existing_header) != schema.header:
            raise ValueError(
                f"Existing header in {path.name} does not match the schema. "
                f"Found: {', '.join(existing_header)}. "
                f"Expected: {', '.join(schema.header)}."
            )
        if row_exists(path, schema, row):
            return False
        write_header = False
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = True

    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=schema.header)
        if write_header:
            writer.writeheader()
        writer.writerow(dict(row))
    return True
