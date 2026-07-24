"""Download everything the experiments read, before running any of them.

    uv run python artifact/setup_assets.py           # everything this install can fetch
    uv run python artifact/setup_assets.py --list    # show the assets and their sizes
    uv run python artifact/setup_assets.py --only celeba,cifar10

This is step one of the reviewer workflow. Every experiment assumes its inputs
are already on disk, and the runtimes quoted in `ARTIFACT.md` and `RUNTIME.md`
are compute time that excludes the downloads performed here. Run this once, let
it finish, and the sweeps that follow neither reach the network nor surprise you
with a multi-gigabyte fetch an hour in.

Two things make it worth a separate step rather than a side effect of whichever
run first needs a file.

*Concurrency.* Level 3 fans out across nodes sharing one `data/` cache. Left to
themselves, several nodes reaching a dataset for the first time race the same
download into the same directory. Serialising it here means every later run
finds the cache warm and reads it.

*Failing fast.* A gated model repository, an expired token or a broken mirror
surfaces here in minutes rather than hours into a sweep.

Nothing here trains anything: model weights are downloaded, not fitted. Assets
are fetched exactly as the experiments fetch them, through the same loaders and
the same Hugging Face cache, so an asset this script reports as ready is one the
experiments can read.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from common.paths import repo_root


@dataclass(frozen=True)
class Asset:
    """One downloadable input, and which experiments need it.

    Attributes:
        name: The name this script and `--only` know the asset by.
        needed_by: Comma-separated experiment ids, for the progress line.
        size: Approximate download size, so a reviewer can plan disk and time.
        fetch: Callable performing the download, returning a one-line detail
            string. Raising is how it reports failure.
        needs_llm: Whether the asset requires the optional `llm` extra. Assets
            that do are skipped with a note when the extra is absent, because
            E1-E4 do not need it.
    """

    name: str
    needed_by: str
    size: str
    fetch: Callable[[], str]
    needs_llm: bool = False


def _fetch_dataset(name: str) -> Callable[[], str]:
    """Return a fetcher loading one `amulet.utils.load_data` dataset."""

    def fetch() -> str:
        from amulet.utils import load_data

        data = load_data(repo_root(), name)
        train = cast("Subset[object]", data.train_set)
        test = cast("Subset[object]", data.test_set)
        return f"{len(train)} train / {len(test)} test"

    return fetch


def _fetch_sst2() -> str:
    """Pull SST-2 into the same `data/sst2` cache E5 reads."""
    from datasets import load_dataset

    cache_dir = str(repo_root() / "data" / "sst2")
    sizes: list[str] = []
    for split in ("train", "validation"):
        raw = load_dataset("stanfordnlp/sst2", split=split, cache_dir=cache_dir)
        sizes.append(f"{len(raw)} {split}")
    return " / ".join(sizes)


def _fetch_hf_repo(repo_id: str, *, tokenizer_only: bool = False) -> Callable[[], str]:
    """Return a fetcher downloading a Hugging Face repository into the shared cache.

    Weights are downloaded, never loaded: `snapshot_download` writes the files
    and returns, so this costs no GPU memory and no model-construction time.

    Args:
        repo_id: The Hugging Face repository, e.g. `meta-llama/Llama-3.2-3B`.
        tokenizer_only: Fetch just the tokenizer files. The `test` level builds
            a randomly-initialised model from a config and needs no weights.

    Returns:
        A callable performing the download and reporting where it landed.
    """

    def fetch() -> str:
        from huggingface_hub import snapshot_download

        patterns = (
            ["tokenizer*", "special_tokens_map.json", "*.model"]
            if tokenizer_only
            else None
        )
        path = snapshot_download(repo_id=repo_id, allow_patterns=patterns)
        return f"cached at {path}"

    return fetch


# Every asset the five experiments read at `smoke` or `full`, and which needs it.
# `test` substitutes synthetic tabular data for E1-E4 and downloads nothing for
# them; it does need the TinyLlama tokenizer, which is why that is listed.
ASSETS: tuple[Asset, ...] = (
    Asset("celeba", "E1", "~1.4 GB", _fetch_dataset("celeba")),
    Asset("census", "E2, E3, E4", "~5 MB", _fetch_dataset("census")),
    Asset("lfw", "E2, E3, E4", "~200 MB", _fetch_dataset("lfw")),
    Asset("fmnist", "E2, E4", "~30 MB", _fetch_dataset("fmnist")),
    Asset("cifar10", "E2, E4", "~170 MB", _fetch_dataset("cifar10")),
    Asset("sst2", "E5", "~7 MB", _fetch_sst2, needs_llm=True),
    Asset(
        "llama-3.2-3b",
        "E5",
        "~6.5 GB",
        _fetch_hf_repo("meta-llama/Llama-3.2-3B"),
        needs_llm=True,
    ),
    Asset(
        "tinyllama-tokenizer",
        "E5 (test level)",
        "~5 MB",
        _fetch_hf_repo("TinyLlama/TinyLlama-1.1B-Chat-v1.0", tokenizer_only=True),
        needs_llm=True,
    ),
)

ASSETS_BY_NAME: dict[str, Asset] = {asset.name: asset for asset in ASSETS}


def llm_extra_installed() -> bool:
    """Report whether the optional `llm` extra is importable.

    Returns:
        True if the Hugging Face stack E5 needs is present.
    """
    from importlib.util import find_spec

    return all(find_spec(name) is not None for name in ("transformers", "datasets"))


def _print_listing() -> None:
    """Print the asset table without downloading anything."""
    have_llm = llm_extra_installed()
    print(f"{'asset':22} {'needed by':16} {'size':>9}  note")
    for asset in ASSETS:
        note = ""
        if asset.needs_llm:
            note = "needs `llm` extra" + ("" if have_llm else " (NOT INSTALLED)")
        print(f"{asset.name:22} {asset.needed_by:16} {asset.size:>9}  {note}")


def _select(requested: str) -> tuple[list[Asset], list[str]]:
    """Resolve the `--only` argument into assets to fetch.

    Args:
        requested: `all`, or a comma-separated list of asset names.

    Returns:
        The selected assets, and any requested names that matched nothing.
    """
    if requested.strip() == "all":
        return list(ASSETS), []
    names = [piece.strip() for piece in requested.split(",") if piece.strip()]
    unknown = [name for name in names if name not in ASSETS_BY_NAME]
    return [ASSETS_BY_NAME[n] for n in names if n in ASSETS_BY_NAME], unknown


def main(argv: list[str] | None = None) -> int:
    """Download the requested assets from the command line.

    Returns:
        Process exit code: 0 if every selected asset is ready, 1 if any failed.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only",
        type=str,
        default="all",
        help=f"Comma-separated subset of: {', '.join(ASSETS_BY_NAME)}. Default: all.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the assets, their sizes and what needs them, then exit.",
    )
    args = parser.parse_args(argv)

    if args.list:
        _print_listing()
        return 0

    selected, unknown = _select(args.only)
    if unknown:
        print(f"Unknown asset(s): {', '.join(unknown)}.")
        print(f"Known: {', '.join(ASSETS_BY_NAME)}.")
        return 1

    have_llm = llm_extra_installed()
    print(f"fetching {len(selected)} asset(s); datasets land in {repo_root() / 'data'}")
    if not have_llm:
        print("`llm` extra not installed: E5's assets will be skipped\n")
    else:
        print()

    failed: list[str] = []
    skipped: list[str] = []
    for asset in selected:
        if asset.needs_llm and not have_llm:
            skipped.append(asset.name)
            print(
                f"  {asset.name:22} ({asset.needed_by:16}) ... skipped (no llm extra)"
            )
            continue
        print(f"  {asset.name:22} ({asset.needed_by:16}) ... ", end="", flush=True)
        started = time.perf_counter()
        try:
            detail = asset.fetch()
        except Exception as exception:
            print(f"FAILED  {type(exception).__name__}: {exception}")
            failed.append(asset.name)
            continue
        print(f"ok  {detail}  [{time.perf_counter() - started:.1f}s]")

    print()
    if failed:
        print(f"{len(failed)} asset(s) failed: {', '.join(failed)}")
        if any(ASSETS_BY_NAME[name].needs_llm for name in failed):
            print(
                "Hugging Face repositories can be gated: accept the model licence on "
                "the hub and authenticate with `hf auth login` before retrying."
            )
        return 1
    if skipped:
        print(
            f"{len(skipped)} E5 asset(s) skipped: {', '.join(skipped)}. "
            "Install the extra (`uv sync --extra cu128 --extra llm`) and re-run "
            "to fetch them."
        )
    print("every selected asset is on disk; the experiments can now run offline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
