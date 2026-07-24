"""Tiny end-to-end run of E4, Outlier Removal x Model Ownership (plan S8, L1).

At `test` level a small dense net over a handful of synthetic tabular rows
stands in for every dataset, so the whole pipeline (load, split, train the clean
baseline through the shared cache, purify via kNN-Shapley outlier removal and
retrain the defended model, distil a surrogate, score fidelity, append a row)
runs on CPU in seconds with no download. kNN-Shapley is O(train x test), so the
tiny stand-in (64 train / 32 test rows) keeps it sub-second. What is asserted is
"each cell produces a well-formed row with finite, in-range numbers,
reproducibly", never paper accuracy: E4 is a reconstruction with no ground-truth
CSV (plan S13.3), so numeric reproduction waits on an L3 run.

Two load-bearing tests here pin the plan's decisions:
* `test_the_baseline_reuses_e2s_clean_checkpoint` proves the E2<->E4 shared
  baseline (plan S6, S13) at runtime: the file E4's clean baseline loads is the
  one E2 already wrote.
* `test_outlier_removal_produces_a_distinct_defended_model` proves a removed
  percentage is a genuinely different, retrained model, not the clean baseline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path

# Every metric E4 records is a percentage in [0, 100].
_PERCENT_COLUMNS = (
    "defended_test_acc",
    "stolen_test_acc",
    "fidelity",
    "correct_fidelity",
)


def _context(tmp_path: Path, seed: int = 0):
    """Build a `test`-level run context over a throwaway cache.

    Wires E4's outlier-carrying tiny data factory, matching what `run` does, so a
    context built directly here exercises the same data path as the CLI.
    """
    from common.config import get_level
    from experiments import shared_targets as targets
    from experiments.e4_outrem_modext.run import tiny_outrem_dataset

    config = get_level("test").with_defaults(epochs=100)
    torch.set_num_threads(1)
    targets.seed_everything(seed)
    return targets.RunContext(
        level=config,
        seed=seed,
        device="cpu",
        cache_dir=tmp_path / "models",
        tiny_data_factory=tiny_outrem_dataset,
    )


@pytest.mark.integration
def test_test_level_run_writes_an_in_range_row(tmp_path: Path) -> None:
    """One cell produces a single CSV row whose metrics are all in [0, 100]."""
    from common.io import read_rows
    from experiments.e4_outrem_modext import run as e4

    rows = e4.run(
        level="test",
        seeds=(0,),
        datasets=("census",),
        percents=(10,),
        output_dir=tmp_path,
    )

    assert len(rows) == 1
    for column in _PERCENT_COLUMNS:
        value = cast(float, rows[0][column])
        assert 0.0 <= value <= 100.0, f"{column} outside [0, 100]: {value}"
    assert len(read_rows(tmp_path / "e4_outrem_modext.csv")) == 1


@pytest.mark.integration
def test_the_baseline_percent_is_the_clean_model(tmp_path: Path) -> None:
    """At percent 0 the defended model *is* the clean baseline, not a retrain.

    The zero-removal case is the table's $\\modelstd$ column and the figures'
    leftmost point: no outliers are removed, so no retraining happens and the
    defended model is the same object and checkpoint as the clean baseline.
    """
    from experiments.e4_outrem_modext import run as e4

    ctx = _context(tmp_path)
    bundle, _ = e4.build_models(ctx, "census", 0)

    assert bundle.defended is bundle.clean
    assert bundle.defended_spec.key() == bundle.clean_spec.key()


@pytest.mark.integration
def test_outlier_removal_produces_a_distinct_defended_model(tmp_path: Path) -> None:
    """A removed percentage retrains a distinct network from the clean baseline.

    The defended model is a separate object, hashes to a separate checkpoint,
    and holds different weights, so the "defended" measurements at 10%+ removal
    cannot secretly be the clean baseline's.
    """
    from experiments.e4_outrem_modext import run as e4

    ctx = _context(tmp_path)
    bundle, _ = e4.build_models(ctx, "census", 10)

    assert bundle.defended is not bundle.clean
    assert bundle.defended_spec.key() != bundle.clean_spec.key()

    clean_weights = list(bundle.clean.state_dict().values())
    defended_weights = list(bundle.defended.state_dict().values())
    assert any(
        not torch.equal(a, b)
        for a, b in zip(clean_weights, defended_weights, strict=True)
    ), "outlier removal left the weights identical to the clean baseline"


@pytest.mark.integration
def test_the_baseline_reuses_e2s_clean_checkpoint(tmp_path: Path) -> None:
    """E4's clean baseline loads the checkpoint E2 already wrote (plan S6, S13).

    Both experiments describe the clean model-extraction target with the
    identical spec (the 50/50 dataset-level split, Adam at 1e-3, matching epochs
    and batch), so its content hash coincides. Running E2 first, then asking for
    E4's baseline spec, the file E4 would load already exists: one checkpoint,
    two experiments. Were the specs to diverge, this file would be absent.
    """
    from common.models import checkpoint_path
    from experiments import shared_targets as targets
    from experiments.e2_advtr_modext import run as e2
    from experiments.e4_outrem_modext import run as e4

    cache = tmp_path / "shared_models"

    _ = e2.run(
        level="test",
        seeds=(0,),
        datasets=("census",),
        epsilons=(0.01,),
        output_dir=tmp_path / "e2_out",
        cache_dir=cache,
    )

    ctx = _context(tmp_path)
    ctx.cache_dir = cache
    data = ctx.data("census")
    batch_size = targets.batch_for(ctx.level, e4.BATCH_SIZE)
    e4_clean_spec = e4.clean_baseline_spec(
        ctx, "census", data.num_features, data.num_classes, batch_size
    )

    assert checkpoint_path(e4_clean_spec, cache_dir=cache).exists()


@pytest.mark.integration
def test_the_clean_baseline_is_trained_once_across_percentages(
    tmp_path: Path,
) -> None:
    """Two removal percentages share one clean checkpoint (plan S6).

    Sweeping percents (0, 10) leaves four checkpoints: one shared clean baseline,
    one outlier-removed defended model (10% only; 0% reuses the baseline), and
    one stolen surrogate per percentage. Were the clean target keyed on the
    percentage, it would train twice and leave five.
    """
    from experiments.e4_outrem_modext import run as e4

    _ = e4.run(
        level="test",
        seeds=(0,),
        datasets=("census",),
        percents=(0, 10),
        output_dir=tmp_path,
        cache_dir=tmp_path / "models",
    )

    assert len(sorted((tmp_path / "models").glob("*.pt"))) == 4


@pytest.mark.integration
def test_defended_test_acc_is_measured_on_the_defended_model(tmp_path: Path) -> None:
    """The row's defended test accuracy is the defended model's, on the test set."""
    from amulet.utils import get_accuracy
    from experiments import shared_targets as targets
    from experiments.e4_outrem_modext import run as e4

    ctx = _context(tmp_path)
    bundle, data = e4.build_models(ctx, "census", 10)
    test_loader = targets.loader_for(data.test_set, targets.batch_for(ctx.level, 256))

    row = e4.run_cell(_context(tmp_path), "census", 10, tmp_path / "out")[0]

    assert cast(float, row["defended_test_acc"]) == pytest.approx(
        get_accuracy(bundle.defended, test_loader, "cpu")
    )


@pytest.mark.integration
def test_rerunning_into_the_same_csv_does_not_duplicate(tmp_path: Path) -> None:
    """The idempotent append resumes an interrupted sweep rather than doubling it."""
    from common.io import read_rows
    from experiments.e4_outrem_modext import run as e4

    for _ in range(2):
        _ = e4.run(
            level="test",
            seeds=(0,),
            datasets=("census",),
            percents=(10,),
            output_dir=tmp_path,
            cache_dir=tmp_path / "models",
        )

    assert len(read_rows(tmp_path / "e4_outrem_modext.csv")) == 1


@pytest.mark.integration
def test_same_seed_reproduces(tmp_path: Path) -> None:
    """Two seeded runs agree numerically, so a reviewer's re-run is comparable.

    Agreement is to a tolerance, not bitwise: CPU float reductions are not
    bit-reproducible once the intra-op pool is warm. A swapped or mis-scaled
    metric moves a cell far more than this.
    """
    from experiments.e4_outrem_modext import run as e4

    def once(where: Path) -> dict[str, object]:
        return e4.run(
            level="test",
            seeds=(0,),
            datasets=("census",),
            percents=(10,),
            output_dir=where / "out",
            cache_dir=where / "models",
        )[0]

    first = once(tmp_path / "a")
    second = once(tmp_path / "b")

    for column in _PERCENT_COLUMNS:
        assert first[column] == pytest.approx(second[column], abs=1.0), column


@pytest.mark.integration
def test_a_test_level_run_leaves_the_committed_results_untouched(
    tmp_path: Path,
) -> None:
    """A tiny run must never write where the paper's numbers live."""
    from common.io import results_path
    from experiments.e4_outrem_modext import run as e4

    committed = results_path("e4_outrem_modext")
    before = committed.read_text() if committed.exists() else None

    _ = e4.run(level="test", seeds=(0,), datasets=("census",), percents=(10,))

    after = committed.read_text() if committed.exists() else None
    assert after == before
