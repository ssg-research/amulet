"""Contract for the registry-driven experiment runner (plan §9, P5).

`run_experiments.py` is the top of the wrapper hierarchy: it iterates the
experiment registry and drives each experiment's uniform `run(level, seeds)`.
These tests pin the dispatch — every registered experiment is invoked, with the
requested level and seeds — by mocking `run()` at the `load_experiment` boundary,
so they stay on the fast tier (no training). A tiny end-to-end run at `test`
level lives in the integration suite.
"""

from __future__ import annotations

import pytest
import run_experiments
from run_experiments import parse_only, parse_seeds
from run_experiments import run_experiments as run_all

from common.registry import EXPERIMENT_IDS


class _RecordingModule:
    """A stand-in experiment module whose `run` records how it was dispatched."""

    def __init__(self, experiment_id: str, log: list[tuple[str, str, object]]) -> None:
        self._id = experiment_id
        self._log = log

    def run(self, level: str, seeds: object) -> list[dict[str, int]]:
        self._log.append((self._id, level, seeds))
        # Two rows, so the reported row count is checkable and non-trivial.
        return [{"row": 1}, {"row": 2}]


def test_dispatches_to_every_registered_experiment(monkeypatch) -> None:
    """Every experiment in the registry is invoked, in registry order.

    Mocking at `load_experiment` keeps the test on real dispatch logic while
    replacing only the heavy training boundary.
    """
    dispatched: list[tuple[str, str, object]] = []
    monkeypatch.setattr(
        run_experiments,
        "load_experiment",
        lambda experiment_id: _RecordingModule(experiment_id, dispatched),
    )

    results = run_all(level="test", seeds=(0,))

    assert [call[0] for call in dispatched] == list(EXPERIMENT_IDS)
    assert [result.experiment_id for result in results] == list(EXPERIMENT_IDS)
    assert all(result.ok for result in results)
    assert all(result.rows == 2 for result in results)


def test_passes_the_requested_level_and_seeds_through(monkeypatch) -> None:
    """The level and seeds reach every runner unchanged."""
    dispatched: list[tuple[str, str, object]] = []
    monkeypatch.setattr(
        run_experiments,
        "load_experiment",
        lambda experiment_id: _RecordingModule(experiment_id, dispatched),
    )

    _ = run_all(level="smoke", seeds=(0, 3))

    assert all(level == "smoke" and seeds == (0, 3) for _, level, seeds in dispatched)


def test_only_subset_runs_just_those_experiments(monkeypatch) -> None:
    """`only` restricts the sweep to the named experiments, in registry order."""
    dispatched: list[tuple[str, str, object]] = []
    monkeypatch.setattr(
        run_experiments,
        "load_experiment",
        lambda experiment_id: _RecordingModule(experiment_id, dispatched),
    )

    results = run_all(level="test", only=("e4_outrem_modext", "e1_attack_baselines"))

    # parse_only is not applied inside run_experiments(); the caller passes the
    # tuple as given, so order here is the caller's order.
    assert {result.experiment_id for result in results} == {
        "e1_attack_baselines",
        "e4_outrem_modext",
    }


def test_a_failing_runner_is_reported_not_raised(monkeypatch) -> None:
    """One experiment raising is captured as a FAIL, not an abort of the sweep."""

    class _Boom:
        @staticmethod
        def run(level: str, seeds: object) -> list[dict[str, int]]:
            raise RuntimeError("simulated training failure")

    monkeypatch.setattr(run_experiments, "load_experiment", lambda _id: _Boom())

    results = run_all(level="test", only=("e1_attack_baselines", "e2_advtr_modext"))

    assert len(results) == 2
    assert all(not result.ok for result in results)
    assert all(
        "simulated training failure" in (result.error or "") for result in results
    )


def test_parse_seeds_handles_ranges_and_lists() -> None:
    """Seed parsing accepts `0`, `0-4` and `0,2,3`, de-duplicating in order."""
    assert parse_seeds("0") == (0,)
    assert parse_seeds("0-4") == (0, 1, 2, 3, 4)
    assert parse_seeds("0,2,3") == (0, 2, 3)


def test_parse_only_validates_against_the_registry() -> None:
    """`--only` accepts registry IDs and `all`, and rejects the unknown."""
    assert parse_only("all") == EXPERIMENT_IDS
    assert parse_only("e1_attack_baselines,e5_textbadnets") == (
        "e1_attack_baselines",
        "e5_textbadnets",
    )
    with pytest.raises(ValueError, match="Unknown experiment"):
        _ = parse_only("e1_attack_baselines,does_not_exist")
