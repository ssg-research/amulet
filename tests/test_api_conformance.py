"""Executable guard for Amulet's defense entry-point convention.

The API contract (see ``AGENTS.md`` / ``docs/CONTRIBUTING.md``) is that every defense
exposes a training-shaped entry-point method for its risk: poisoning/evasion defenses
``train_robust``, membership-inference ``train_private``, fairness ``train_fair``, and the
ownership defenses ``watermark`` / ``fingerprint``. A defense that ships only a bespoke
method (as ONION once did with ``purify``) silently breaks the shared interface.

Documentation stating the rule was not enough — a defense landed without ``train_robust``
anyway — so this test enforces it mechanically: it fails CI regardless of whether anyone
reads the docs. Extra public helpers (e.g. ONION's ``purify``) are fine; the sanctioned
entry point must be present *in addition*.

Adding a defense whose entry point is none of the sanctioned methods is a deliberate
convention change: it must update ``RISK_DEFENSE_ENTRY_POINTS`` here, which is the point at
which a human reconsiders whether the new shape really belongs.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

# Sanctioned entry-point methods per risk. A concrete defense must expose at least one.
RISK_DEFENSE_ENTRY_POINTS: dict[str, set[str]] = {
    "poisoning": {"train_robust"},
    "evasion": {"train_robust"},
    "membership_inference": {"train_private"},
    "discriminatory_behavior": {"train_fair"},
    "unauth_model_ownership": {"watermark", "fingerprint"},
}


def _concrete_defenses(risk: str) -> list[tuple[str, type]]:
    """Return the ``(name, class)`` of every concrete defense a risk's package exports.

    Base ABCs are abstract (they carry ``abstractmethod``s), so ``inspect.isabstract``
    filters them out, leaving only the instantiable defenses the contract governs.
    """
    module = importlib.import_module(f"amulet.{risk}.defenses")
    exported = getattr(module, "__all__", [])
    defenses: list[tuple[str, type]] = []
    for name in exported:
        obj = getattr(module, name)
        if inspect.isclass(obj) and not inspect.isabstract(obj):
            defenses.append((name, obj))
    return defenses


def _all_concrete_defenses() -> list[tuple[str, str, type]]:
    cases: list[tuple[str, str, type]] = []
    for risk in RISK_DEFENSE_ENTRY_POINTS:
        for name, cls in _concrete_defenses(risk):
            cases.append((risk, name, cls))
    return cases


def test_every_risk_has_at_least_one_concrete_defense():
    """Guards the discovery itself: a risk silently exporting no defense would make the
    parametrized test below vacuously pass, so assert every governed risk has one."""
    for risk in RISK_DEFENSE_ENTRY_POINTS:
        assert _concrete_defenses(risk), f"no concrete defenses discovered for {risk}"


@pytest.mark.parametrize(
    ("risk", "name", "cls"),
    _all_concrete_defenses(),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_defense_exposes_sanctioned_entry_point(risk: str, name: str, cls: type):
    """Every concrete defense exposes a sanctioned training entry point for its risk."""
    sanctioned = RISK_DEFENSE_ENTRY_POINTS[risk]
    present = {m for m in sanctioned if callable(getattr(cls, m, None))}
    assert present, (
        f"{name} ({risk}) exposes none of the sanctioned entry-point methods "
        f"{sorted(sanctioned)}. Every defense must implement its risk's training entry "
        f"point (see AGENTS.md); a bespoke method alone breaks the shared interface."
    )
