"""Result-CSV schema and sweep grid for E4 (Outlier Removal x Model Ownership).

Kept free of torch imports so the table and plot renderers in `make/` can
validate a CSV header without loading a model (plan S7.1).

One CSV, `results/e4_outrem_modext.csv`, feeds *both* the table renderer and the
two-figure plot renderer. A row records one $(\\text{dataset}, \\text{seed},
\\text{percent})$ cell: the defended model's test accuracy, and the surrogate
stolen from it (its test accuracy, its fidelity to $\\modeldef$, and its correct
fidelity). `percent` is the fraction of training outliers removed before
retraining, `0` being the clean baseline $\\modelstd$ (no removal), which is the
same checkpoint E2 trains for its clean baseline (see `run.py`).

The reference table lays the removal grid out as columns ($\\modelstd$, 10%, 20%,
30%, 40%) and the figures lay it out as the x-axis (0, 10, 20, 30, 40), so both
renderers read `percent` the same way: `0` is the leftmost point / the
$\\modelstd$ column.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.io import CsvSchema

EXPERIMENT_ID = "e4_outrem_modext"

# Datasets in the reference table's block order (census, lfw, fmnist, cifar),
# which is also the figures' legend order. This differs from E2's block order.
DATASETS: tuple[str, ...] = ("census", "lfw", "fmnist", "cifar")

# Fractions of training outliers removed before retraining, from
# the paper evaluation section / `tab_outrem_modext`. `0` is the clean baseline: no outlier
# removal, so $\\modeldef$ is just $\\modelstd$ and the table's first column.
PERCENTS: tuple[int, ...] = (0, 10, 20, 30, 40)

# The single capacity the reference table reports (VGG11-equivalent `m1`).
CAPACITY = "m1"

SCHEMA = CsvSchema(
    header=(
        "exp_id",
        "dataset",
        "arch",
        "capacity",
        "training_size",
        "epochs",
        "batch_size",
        "adv_train_fraction",
        "percent",
        # Defended model ($\\modeldef$) clean test accuracy. At `percent == 0`
        # this is the clean baseline $\\modelstd$'s test accuracy.
        "defended_test_acc",
        # Stolen surrogate ($\\modelstol$), distilled from and scored against the
        # model at this removal percentage.
        "stolen_test_acc",
        "fidelity",
        "correct_fidelity",
        # Wall-clock cost of the cell, measured from just after the resume check
        # to just before the row is written. What `RUNTIME.md` is built from.
        # This is real work only when the checkpoint cache is cold: the CSV
        # resume check and `.model_cache/` are independent, so a row absent from
        # the CSV is still written after merely *loading* models it has trained
        # before, and records seconds instead of hours.
        "runtime_sec",
        "timestamp",
    ),
    key_columns=("exp_id", "dataset", "capacity", "percent"),
)
