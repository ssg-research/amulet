"""Result-CSV schemas for E5's two sub-experiments.

Kept in their own module, free of torch and Hugging Face imports, so the table
renderer in `make/` can validate a CSV header without loading a 3B victim.

The headers are those the paper run wrote, column for column, so the committed
CSVs under `artifact/results/e5_textbadnets/` are readable as-is. The key
columns identify an experiment *cell*: which measurement a row is, not what it
measured. Re-running a completed sweep therefore appends nothing (plan §7.1).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.io import CsvSchema

ONION_SCHEMA = CsvSchema(
    header=(
        "exp_id",
        "dataset",
        "model_name",
        "reference_model",
        "dtype",
        "num_classes",
        "max_length",
        "n_train",
        "clean_test_size",
        "asr_test_size",
        "batch_size",
        "epochs",
        "lr",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "trigger",
        "trigger_label",
        "insert_position",
        "poisoned_portion",
        "n_poisoned_train",
        "onion_threshold",
        "clean_baseline_test_acc",
        "undef_test_acc",
        "undef_asr",
        "def_test_acc_purified",
        "def_test_acc_raw",
        "def_asr",
        "trigger_removal_rate",
        "mean_words_removed",
        "clean_train_runtime_sec",
        "undef_train_runtime_sec",
        "def_train_runtime_sec",
        "onion_purify_runtime_sec",
        "timestamp",
    ),
    key_columns=("exp_id", "poisoned_portion"),
)

DP_SCHEMA = CsvSchema(
    header=(
        "exp_id",
        "dataset",
        "model_name",
        "dtype",
        "num_classes",
        "max_length",
        "n_train",
        "clean_test_size",
        "asr_test_size",
        "batch_size",
        "epochs",
        "lr",
        "dp_epochs",
        "dp_lr",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "trigger",
        "trigger_label",
        "insert_position",
        "poisoned_portion",
        "n_poisoned_train",
        "clean_baseline_test_acc",
        "undef_test_acc",
        "undef_asr",
        "target_epsilon",
        "epsilon",
        "sigma",
        "delta",
        "max_per_sample_grad_norm",
        "dp_test_acc",
        "dp_asr",
        "clean_train_runtime_sec",
        "undef_train_runtime_sec",
        "dp_train_runtime_sec",
        "timestamp",
    ),
    key_columns=("exp_id", "poisoned_portion", "target_epsilon"),
)
