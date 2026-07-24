"""Verification-level presets shared by every experiment.

The artifact is verifiable at three escalating levels (plan §8):

* `test` — does the script *work*? Tiny data, tiny model, one epoch, CPU.
* `smoke` — is the experiment *sound*? Real architectures, one epoch, a small
  fraction of *both* splits, and every repeated-work loop shrunk to its floor
  (shadow bank, inversion steps, PGD chain, E5's poison-rate grid); minutes on
  one GPU. Only `full` reads a whole split or runs a paper-sized loop, so a knob
  that reduces a level's cost is gated on `train_fraction < 1.0`, not on
  `tiny_model` (which fires only at `test`).
* `full` — do we reproduce the paper? Paper settings, one seed by default.

`full` deliberately leaves `epochs` unset: the paper epoch count differs per
experiment, so baking one global number here would be wrong for four of the
five. An experiment supplies its own via `with_defaults(epochs=...)`, which
fills the field only when the level left it unset, so `test` and `smoke` keep
their one-epoch budget. A CLI flag that must win regardless goes through
`override(...)` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class LevelConfig:
    """Knobs one verification level applies to an experiment run.

    Attributes:
        name: Level name, one of `test`, `smoke` or `full`.
        seeds: Seeds to sweep. One seed by default; the paper means are over
            five, and a reviewer reads a single seed as within the reported std.
        epochs: Training epochs, or None when the level defers to the
            experiment's own paper setting.
        train_fraction: Fraction of the training split to use, passed through to
            `amulet.utils.load_data(training_size=...)`.
        test_fraction: Fraction of the test split to use, passed through to
            `amulet.utils.load_data(test_size=...)`. Shrinking the training
            split alone does not make a level cheap: evaluation walks the test
            split on every cell, and kNN-Shapley outlier removal (E4) costs
            `O(train x test)`, so a full test split pins its cost near the
            paper's however small the training budget gets.
        tiny_model: Whether to substitute a micro-architecture for the real one.
    """

    name: str
    seeds: tuple[int, ...]
    epochs: int | None
    train_fraction: float
    test_fraction: float
    tiny_model: bool

    def with_defaults(self, *, epochs: int) -> LevelConfig:
        """Fill fields this level left unset with experiment-specific settings.

        Args:
            epochs: The experiment's paper epoch count, applied only when the
                level did not pin one.

        Returns:
            A level with every field populated. Fields the preset already pinned
            are returned unchanged.
        """
        if self.epochs is not None:
            return self
        return replace(self, epochs=epochs)

    def override(
        self,
        *,
        seeds: tuple[int, ...] | None = None,
        epochs: int | None = None,
        train_fraction: float | None = None,
        test_fraction: float | None = None,
        tiny_model: bool | None = None,
    ) -> LevelConfig:
        """Return a copy with the supplied fields forced, preset value or not.

        This is the path for explicit user intent, such as `--seeds 0-4` on a
        full run. Arguments left as None keep the current value.

        Args:
            seeds: Replacement seed tuple.
            epochs: Replacement epoch count.
            train_fraction: Replacement training-split fraction.
            test_fraction: Replacement test-split fraction.
            tiny_model: Replacement micro-architecture flag.

        Returns:
            A level with the supplied fields replaced.
        """
        return replace(
            self,
            seeds=self.seeds if seeds is None else seeds,
            epochs=self.epochs if epochs is None else epochs,
            train_fraction=(
                self.train_fraction if train_fraction is None else train_fraction
            ),
            test_fraction=(
                self.test_fraction if test_fraction is None else test_fraction
            ),
            tiny_model=self.tiny_model if tiny_model is None else tiny_model,
        )


LEVELS: dict[str, LevelConfig] = {
    "test": LevelConfig(
        name="test",
        seeds=(0,),
        epochs=1,
        train_fraction=0.01,
        test_fraction=0.01,
        tiny_model=True,
    ),
    "smoke": LevelConfig(
        name="smoke",
        seeds=(0,),
        epochs=1,
        train_fraction=0.1,
        test_fraction=0.1,
        tiny_model=False,
    ),
    "full": LevelConfig(
        name="full",
        seeds=(0,),
        epochs=None,
        train_fraction=1.0,
        test_fraction=1.0,
        tiny_model=False,
    ),
}

# Ordered cheapest first; use as the `choices` of a `--level` argument.
LEVEL_NAMES: tuple[str, ...] = ("test", "smoke", "full")


def get_level(name: str) -> LevelConfig:
    """Look up a verification-level preset by name.

    Args:
        name: Level name, one of `LEVEL_NAMES`.

    Returns:
        The immutable preset for that level.

    Raises:
        ValueError: If `name` is not a known level.
    """
    if name not in LEVELS:
        known = ", ".join(LEVEL_NAMES)
        raise ValueError(f"Unknown level {name!r}. Choose one of: {known}.")
    return LEVELS[name]
