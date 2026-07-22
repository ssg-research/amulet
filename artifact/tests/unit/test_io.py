"""Tests for common/io.py.

Every experiment appends one row per configuration to a CSV under
`artifact/runs/<level>/`, and the `make/` wrappers render from them so they can
render tables with no GPU (plan §13, decision 2). That makes two properties
load-bearing: an interrupted sweep must be resumable without duplicating rows,
and a header that has drifted from the schema must fail loudly rather than
silently writing a misaligned row.
"""

import csv
from pathlib import Path

import pytest

from common.io import (
    CsvSchema,
    append_row,
    default_results_dir,
    read_rows,
    results_path,
)

SCHEMA = CsvSchema(
    header=("dataset", "seed", "epsilon", "accuracy", "fidelity"),
    key_columns=("dataset", "seed", "epsilon"),
)

ROW: dict[str, object] = {
    "dataset": "census",
    "seed": 0,
    "epsilon": 0.1,
    "accuracy": 0.8123,
    "fidelity": 0.7654,
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_the_default_results_dir_is_the_full_run_output() -> None:
    """No CSVs ship, so a renderer given no directory reads a full run's output."""
    from common.paths import artifact_root

    assert default_results_dir() == artifact_root() / "runs" / "full"


def test_results_path_defaults_to_the_experiment_id() -> None:
    assert (
        results_path("e2_advtr_modext") == default_results_dir() / "e2_advtr_modext.csv"
    )


def test_results_path_accepts_a_stem_for_multi_csv_experiments() -> None:
    path = results_path("e5_textbadnets", stem="onion_sst2")
    assert path == default_results_dir() / "e5_textbadnets" / "onion_sst2.csv"


def test_append_row_writes_a_header_then_the_row(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"

    assert append_row(path, SCHEMA, ROW) is True

    assert path.read_text().splitlines()[0] == "dataset,seed,epsilon,accuracy,fidelity"
    assert _rows(path) == [
        {
            "dataset": "census",
            "seed": "0",
            "epsilon": "0.1",
            "accuracy": "0.8123",
            "fidelity": "0.7654",
        }
    ]


def test_append_row_creates_missing_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "out.csv"

    _ = append_row(path, SCHEMA, ROW)

    assert path.is_file()


def test_appending_the_same_row_twice_writes_it_once(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"

    assert append_row(path, SCHEMA, ROW) is True
    assert append_row(path, SCHEMA, ROW) is False

    assert len(_rows(path)) == 1


def test_idempotency_matches_on_key_columns_only(tmp_path: Path) -> None:
    # A resumed run recomputes the same cell; measured values may differ in the
    # last decimal. The key columns identify the cell, so it is still a duplicate.
    path = tmp_path / "out.csv"
    _ = append_row(path, SCHEMA, ROW)

    assert append_row(path, SCHEMA, {**ROW, "accuracy": 0.9}) is False
    assert len(_rows(path)) == 1


def test_numeric_keys_compare_by_value_not_string(tmp_path: Path) -> None:
    # 0.10 and 0.1 render differently but name the same experiment cell.
    path = tmp_path / "out.csv"
    _ = append_row(path, SCHEMA, {**ROW, "epsilon": 0.1})

    assert append_row(path, SCHEMA, {**ROW, "epsilon": 0.10}) is False
    assert append_row(path, SCHEMA, {**ROW, "seed": 0}) is False
    assert len(_rows(path)) == 1


def test_a_different_key_column_value_appends_a_new_row(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    _ = append_row(path, SCHEMA, ROW)

    assert append_row(path, SCHEMA, {**ROW, "seed": 1}) is True
    assert append_row(path, SCHEMA, {**ROW, "epsilon": 0.5}) is True
    assert append_row(path, SCHEMA, {**ROW, "dataset": "lfw"}) is True

    assert len(_rows(path)) == 4


def test_append_row_rejects_a_row_missing_a_column(tmp_path: Path) -> None:
    incomplete = {k: v for k, v in ROW.items() if k != "fidelity"}

    with pytest.raises(ValueError, match="fidelity"):
        _ = append_row(tmp_path / "out.csv", SCHEMA, incomplete)


def test_append_row_rejects_a_row_with_an_unknown_column(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="auc"):
        _ = append_row(tmp_path / "out.csv", SCHEMA, {**ROW, "auc": 0.5})


def test_append_row_rejects_a_file_whose_header_drifted(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    _ = path.write_text("dataset,seed,epsilon,accuracy\ncensus,0,0.1,0.8\n")

    with pytest.raises(ValueError, match="header"):
        _ = append_row(path, SCHEMA, ROW)


def test_schema_rejects_a_key_column_absent_from_the_header() -> None:
    with pytest.raises(ValueError, match="capacity"):
        _ = CsvSchema(header=("dataset", "seed"), key_columns=("dataset", "capacity"))


def test_read_rows_returns_an_empty_list_for_a_missing_file(tmp_path: Path) -> None:
    assert read_rows(tmp_path / "absent.csv") == []


def test_read_rows_returns_the_appended_rows(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    _ = append_row(path, SCHEMA, ROW)
    _ = append_row(path, SCHEMA, {**ROW, "seed": 1})

    rows = read_rows(path)

    assert [row["seed"] for row in rows] == ["0", "1"]


def test_rows_survive_a_reopened_file(tmp_path: Path) -> None:
    # Each append opens the file fresh, mirroring separate sweep processes.
    path = tmp_path / "out.csv"
    for seed in range(3):
        _ = append_row(path, SCHEMA, {**ROW, "seed": seed})

    assert len(_rows(path)) == 3
    assert path.read_text().count("dataset,seed") == 1
