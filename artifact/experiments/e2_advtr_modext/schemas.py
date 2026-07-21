"""Result-CSV schema and sweep grid for E2 (Adversarial Training x Model Ownership).

Kept free of torch imports so the table renderer in `make/` can validate a CSV
header without loading a model (plan S7.1).

One CSV, `results/e2_advtr_modext.csv`, holds every dataset block. A row records
one $(\\text{dataset}, \\text{seed}, \\epsilon)$ cell: the clean baseline's test
accuracy (epsilon-independent, so identical across a seed's rows and pooled once
for the baseline row), the defended model's clean and robust accuracies, the
undefended target's robust accuracy at that budget, and the stolen surrogate's
accuracy and fidelity to the defended model.

The key columns identify a cell, so a resumed sweep appends nothing. `epsilon` is
the swept budget; the clean baseline is not a row of its own because its number
lives in every row's `target_test_acc`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.io import CsvSchema

EXPERIMENT_ID = "e2_advtr_modext"

# Datasets in the reference table's block order (census, fmnist, lfw, cifar).
DATASETS: tuple[str, ...] = ("census", "fmnist", "lfw", "cifar")

# The perturbation budgets swept, from `06evaluation.tex` / `tab_advtr_modext`.
EPSILONS: tuple[float, ...] = (0.01, 0.03, 0.05, 0.10)

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
        "epsilon",
        "step_size",
        "iterations",
        # Clean baseline test accuracy, epsilon-independent: identical across a
        # seed's rows, pooled once per seed for the `Baseline (M_std)` row.
        "target_test_acc",
        # Defended model ($\\modeldef$) clean test accuracy.
        "defended_test_acc",
        # Robust accuracy under PGD at this budget: undefended, then defended.
        "target_robust_acc",
        "defended_robust_acc",
        # Stolen surrogate ($\\modelstol$), distilled from and scored against $\\modeldef$.
        "stolen_test_acc",
        "fidelity",
        "correct_fidelity",
        "timestamp",
    ),
    key_columns=("exp_id", "dataset", "capacity", "epsilon"),
)
