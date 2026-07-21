"""Result-CSV schema and sweep grid for E3 (Adversarial Training x Attribute Inference).

Kept free of torch imports so the renderer in `make/` can validate a header
without loading a model (plan S7.1).

One CSV, `results/e3_advtr_attrinf.csv`. Each dataset contributes a baseline row
(attribute inference against the clean $\\modelstd$) plus one row per budget
(attribute inference against the adversarially-trained $\\modeldef$), matching the
reference table's row structure. The two row kinds are distinguished by
`model_role` and by `epsilon`: the baseline carries the sentinel budget
`BASELINE_EPSILON` and leaves the two robust-accuracy columns blank, since there
is no defended model to perturb.

Both sensitive attributes are inferred at once (census: race and sex; lfw: race
and gender, which the table labels Sex), so every row carries an accuracy and an
AUC for each.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.io import CsvSchema

EXPERIMENT_ID = "e3_advtr_attrinf"

# Datasets in the reference table's block order.
DATASETS: tuple[str, ...] = ("census", "lfw")

# The perturbation budgets swept, from `06evaluation.tex` / `tab_attinf_advrtr`.
EPSILONS: tuple[float, ...] = (0.01, 0.03, 0.06, 0.10)

# The sentinel budget of the clean-baseline row, distinct from every real budget.
BASELINE_EPSILON = 0.0

# The single capacity the reference table reports.
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
        # "baseline" (measured on $\\modelstd$) or "defended" (on $\\modeldef$).
        "model_role",
        # Test accuracy of the model this row measured ($\\modelstd$ or $\\modeldef$).
        "test_acc",
        # Robust accuracy under PGD; blank on the baseline row (no defended model).
        "target_robust_acc",
        "defended_robust_acc",
        # Attribute-inference balanced accuracy (percent) and AUC, per attribute.
        "acc_att_race",
        "auc_race",
        "acc_att_sex",
        "auc_sex",
        # The dataset's two sensitive-column names, for provenance.
        "sensitive_attr_1",
        "sensitive_attr_2",
        "timestamp",
    ),
    key_columns=("exp_id", "dataset", "capacity", "epsilon"),
)
