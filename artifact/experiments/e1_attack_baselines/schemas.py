"""Result-CSV schemas for E1's six sub-experiments.

Kept in their own module, free of torch imports, so the table renderer in
`make/` can validate a CSV header without loading a model (plan §7.1).

Six schemas rather than one wide table: the risks measure genuinely different
quantities, and a single header carrying every metric would leave most of each
row empty. Each sub-experiment therefore writes
`results/e1_attack_baselines/<attack>.csv`, the multi-CSV layout `common.io`
already supports for E5.

Every schema opens with the same block of columns describing *which model* was
measured. Those are not decoration: together with the attack's own knobs they
are exactly the fields that go into the `ModelSpec` behind the row, so a reader
can tell from the CSV alone whether two rows shared a target model.

The key columns identify an experiment *cell*: the seed, the capacity column of
the paper's table, and the attack's own swept knob. Re-running a completed sweep
therefore appends nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.io import CsvSchema

# Columns describing the target model a row measured, shared by all six CSVs.
_LEADING: tuple[str, ...] = (
    "exp_id",
    "dataset",
    "arch",
    "capacity",
    "training_size",
    "celeba_target",
    "optimizer_recipe",
    "epochs",
    "batch_size",
)

# `runtime_sec` is the wall-clock cost of the row, measured from just after the
# resume check to just before the row is written. It is what `RUNTIME.md` is
# built from. Read it with the shared model cache in mind: several attacks reuse
# one target per (seed, capacity), so whichever row trains it pays for it and the
# rows that follow load a checkpoint instead. The per-row times are therefore
# order-dependent, and only their sum over a seed is a stable quantity.
_TRAILING: tuple[str, ...] = ("runtime_sec", "timestamp")

EVASION_SCHEMA = CsvSchema(
    header=(
        *_LEADING,
        "epsilon",
        "step_size",
        "iterations",
        "target_test_acc",
        "robust_acc",
        *_TRAILING,
    ),
    key_columns=("exp_id", "capacity", "epsilon"),
)

POISONING_SCHEMA = CsvSchema(
    header=(
        *_LEADING,
        "poisoned_portion",
        "trigger_label",
        "std_test_acc",
        "std_poison_acc",
        "pois_test_acc",
        "pois_poison_acc",
        *_TRAILING,
    ),
    key_columns=("exp_id", "capacity", "poisoned_portion"),
)

MODEL_EXTRACTION_SCHEMA = CsvSchema(
    header=(
        *_LEADING,
        "adv_train_fraction",
        "loss_type",
        "target_test_acc",
        "stolen_test_acc",
        "fidelity",
        "correct_fidelity",
        *_TRAILING,
    ),
    key_columns=("exp_id", "capacity", "adv_train_fraction"),
)

MEMBERSHIP_INFERENCE_SCHEMA = CsvSchema(
    header=(
        *_LEADING,
        "pkeep",
        "num_shadow",
        "target_train_acc",
        "target_test_acc",
        "offline_bal_acc",
        "offline_auc",
        "offline_tpr_at_1fpr",
        "online_bal_acc",
        "online_auc",
        "online_tpr_at_1fpr",
        *_TRAILING,
    ),
    key_columns=("exp_id", "capacity", "pkeep", "num_shadow"),
)

ATTRIBUTE_INFERENCE_SCHEMA = CsvSchema(
    header=(
        *_LEADING,
        "adv_train_fraction",
        "sensitive_attribute",
        "target_test_acc",
        "attack_bal_acc",
        "attack_auc",
        *_TRAILING,
    ),
    key_columns=("exp_id", "capacity", "adv_train_fraction"),
)

DATA_RECONSTRUCTION_SCHEMA = CsvSchema(
    header=(
        *_LEADING,
        "alpha",
        "target_test_acc",
        "mse_avg",
        "mse_0",
        "mse_1",
        "ssim_avg",
        "ssim_0",
        "ssim_1",
        *_TRAILING,
    ),
    key_columns=("exp_id", "capacity", "alpha"),
)

# Attack ID -> schema. The IDs double as the CSV stems and as the values
# `run.py --attacks` accepts, so there is one list to keep in step.
SCHEMAS: dict[str, CsvSchema] = {
    "evasion": EVASION_SCHEMA,
    "poisoning": POISONING_SCHEMA,
    "model_extraction": MODEL_EXTRACTION_SCHEMA,
    "membership_inference": MEMBERSHIP_INFERENCE_SCHEMA,
    "attribute_inference": ATTRIBUTE_INFERENCE_SCHEMA,
    "data_reconstruction": DATA_RECONSTRUCTION_SCHEMA,
}

ATTACKS: tuple[str, ...] = tuple(SCHEMAS)

# The paper's table has one column per VGG capacity, labelled VGG11/13/16/19.
CAPACITIES: tuple[str, ...] = ("m1", "m2", "m3", "m4")

# Membership inference is reported for the first column only: it attacks an
# intentionally overfit ResNet rather than the VGG the other rows share, so the
# remaining columns are blank in the paper and must stay blank here.
SINGLE_COLUMN_ATTACKS: frozenset[str] = frozenset({"membership_inference"})
