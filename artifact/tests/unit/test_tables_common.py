"""Contract for the shared interaction-table aggregation helpers.

Pure functions over CSV-shaped dicts: the seed-pooling and blank-handling both
E2 and E3 rely on, tested once here rather than through each renderer.
"""

from __future__ import annotations

import pytest
from make.tables_common import (
    BLANK,
    cell,
    format_cell,
    mean_and_standard_error,
    pooled_by_seed,
)


def test_pooled_by_seed_returns_one_value_per_seed_in_order() -> None:
    """Values are collected one per seed, ordered by seed."""
    rows = [
        {"exp_id": "1", "fidelity": "95.0"},
        {"exp_id": "0", "fidelity": "94.0"},
    ]

    assert pooled_by_seed(rows, "fidelity") == [94.0, 95.0]


def test_pooled_by_seed_deduplicates_a_repeated_seed() -> None:
    """A seed appearing twice for a column contributes one value, not two.

    The baseline pools an epsilon-independent accuracy from every one of a
    dataset's rows, so the same seed's identical value recurs and must collapse.
    """
    rows = [
        {"exp_id": "0", "target_test_acc": "82.0"},
        {"exp_id": "0", "target_test_acc": "82.0"},
        {"exp_id": "1", "target_test_acc": "84.0"},
    ]

    assert pooled_by_seed(rows, "target_test_acc") == [82.0, 84.0]


def test_pooled_by_seed_skips_blank_cells() -> None:
    """A blank cell (the E3 baseline's robust columns) never enters a mean."""
    rows = [
        {"exp_id": "0", "defended_robust_acc": ""},
        {"exp_id": "1", "defended_robust_acc": "81.0"},
    ]

    assert pooled_by_seed(rows, "defended_robust_acc") == [81.0]


def test_cell_is_a_dash_when_nothing_backs_it() -> None:
    """A column with no data renders as a dash, not a crash or a zero."""
    assert cell([], "fidelity") == BLANK
    assert cell([{"exp_id": "0", "fidelity": ""}], "fidelity") == BLANK


def test_cell_renders_mean_and_error_over_the_seeds_present() -> None:
    """A multi-seed cell is `mean $\\pm$ SE`; {94, 96} -> 95.00 $\\pm$ 1.00."""
    rows = [
        {"exp_id": "0", "fidelity": "94.0"},
        {"exp_id": "1", "fidelity": "96.0"},
    ]

    assert cell(rows, "fidelity") == "95.00~$\\pm$~1.00"


def test_an_empty_cell_cannot_be_formatted() -> None:
    """Formatting no measurements is a bug, so it raises rather than guessing."""
    with pytest.raises(ValueError, match="empty cell"):
        _ = format_cell([])


def test_a_single_measurement_has_zero_standard_error() -> None:
    """One seed has no spread to report."""
    assert mean_and_standard_error([57.17]) == (57.17, 0.0)
