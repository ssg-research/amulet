"""Progress output for the experiment sweeps, so slow is distinguishable from hung.

The amulet library streams per-epoch training loss, but nothing says which
experiment or cell those epochs belong to, so a long run looks the same as a
stuck one. These helpers add that context: `cells` wraps a sweep in a tqdm bar
(one per experiment), and `banner`/`log` print short status lines. Every message
goes through `tqdm.write`, so it renders cleanly above an active bar; everything
writes to stderr, tqdm's default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast

from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

_T = TypeVar("_T")


def cells(items: Iterable[_T], experiment_id: str) -> Iterator[_T]:
    """Iterate a sweep's cells behind a tqdm bar labelled with the experiment.

    Args:
        items: The cells to sweep. Consumed into a list so the bar knows its total.
        experiment_id: The bar's label, e.g. `"e2_advtr_modext"`.

    Returns:
        An iterator over `items` that advances a progress bar per step.
    """
    # disable=None hides the animated bar when stderr is not a TTY (a redirected
    # log or a captured test), so those keep just the plain per-cell lines from
    # `log`; an interactive reviewer still gets the live bar.
    bar = tqdm(
        list(items), desc=experiment_id, unit="cell", dynamic_ncols=True, disable=None
    )
    return cast("Iterator[_T]", bar)


def banner(message: str) -> None:
    """Print a blank line then a delimited banner, above any active bar."""
    tqdm.write("")
    tqdm.write(f"=== {message} ===")


def log(message: str) -> None:
    """Print one status line that coexists with an active tqdm bar."""
    tqdm.write(message)
