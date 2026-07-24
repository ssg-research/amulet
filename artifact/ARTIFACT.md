# Amulet Benchmark Artifact

This `artifact/` directory reproduces the tables and plots in the Amulet
benchmark submission.
Amulet (`amuletml` on PyPI, imported as `amulet`) is a Python library for
evaluating unintended interactions among machine-learning defenses and risks
across security, privacy, and fairness.

No result data ships with this repository.
Every number is produced by running an experiment here, and is compared against
the corresponding table or figure in the paper.

## Setup

Dependencies are pinned and managed with `uv` (never `pip` or `conda`).
Pick one torch build extra matching the host, and add `dev` for the checks.
The text study (E5) additionally needs the `llm` extra.

```bash
uv sync --extra cpu --extra dev            # GPU-less / CI; enough for the tests
uv sync --extra cu128 --extra dev          # CUDA 12.x host
uv sync --extra cu130 --extra dev          # CUDA 13 host
uv sync --extra cu128 --extra dev --extra llm   # add the text/LLM stack for E5
```

Do not request more than one torch extra at once: `cpu`, `cu128`, and `cu130`
are declared conflicting and requesting several errors.
Training at smoke or full scale requires a GPU.

Then download the datasets and model weights every experiment reads:

```bash
uv run python artifact/setup_assets.py           # fetch everything this install can
uv run python artifact/setup_assets.py --list    # what it will fetch, and how large
```

This is a required step, not an optimisation.
Every runtime quoted in this document and in [`RUNTIME.md`](RUNTIME.md) is
compute time and assumes the downloads are already complete.
Doing it up front also keeps parallel runs from racing each other into the same
`data/` cache, and surfaces a gated model repository or an expired token in
minutes rather than hours into a sweep.

The script skips E5's assets when the `llm` extra is absent, so an E1-E4
reviewer needs nothing extra.
E5's Llama weights are gated on the Hugging Face hub: accept the licence and
run `hf auth login` before fetching them.

## Reproducing the paper

Three steps: run an experiment, find its CSV, render the table or plot.

### Step 1: run an experiment

Each experiment is one script with the same interface.
`--level full` is the paper's settings.

```bash
uv run python artifact/experiments/e1_attack_baselines/run.py --level full
uv run python artifact/experiments/e2_advtr_modext/run.py     --level full
uv run python artifact/experiments/e3_advtr_attrinf/run.py    --level full
uv run python artifact/experiments/e4_outrem_modext/run.py    --level full
uv run python artifact/experiments/e5_textbadnets/run.py      --level full
```

Every runner accepts `--level {test,smoke,full}` and `--seeds` (`0`, or `0-4`
for the paper's five-seed means).
Each also takes the knobs its own sweep varies, so a single cell can be run on
its own:

| Experiment | Script                                            | Sweep knobs               | Studies                                    |
| ---------- | ------------------------------------------------- | ------------------------- | ------------------------------------------ |
| E1         | `artifact/experiments/e1_attack_baselines/run.py` | `--attacks --capacities`  | baseline attack per risk                   |
| E2         | `artifact/experiments/e2_advtr_modext/run.py`     | `--datasets --epsilons`   | adversarial training x model ownership     |
| E3         | `artifact/experiments/e3_advtr_attrinf/run.py`    | `--datasets --epsilons`   | adversarial training x attribute inference |
| E4         | `artifact/experiments/e4_outrem_modext/run.py`    | `--datasets --percents`   | outlier removal x model ownership          |
| E5         | `artifact/experiments/e5_textbadnets/run.py`      | `--which {onion,dp,both}` | text backdoor x ONION and x DP-SGD         |

```bash
# One dataset, one budget, one seed.
uv run python artifact/experiments/e2_advtr_modext/run.py --level full --datasets census --epsilons 0.01 --seeds 0
```

Runs are resumable: a sweep that dies halfway is restarted with the same
command, and cells already in the CSV are skipped rather than recomputed.

### Step 2: read the results

A run appends one row per configuration to a CSV under `artifact/runs/<level>/`.
A `--level full` run writes to `artifact/runs/full/`:

| Experiment | CSV written                                                                                                                                    |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| E1         | `artifact/runs/full/e1_attack_baselines/{evasion,poisoning,model_extraction,membership_inference,attribute_inference,data_reconstruction}.csv` |
| E2         | `artifact/runs/full/e2_advtr_modext.csv`                                                                                                       |
| E3         | `artifact/runs/full/e3_advtr_attrinf.csv`                                                                                                      |
| E4         | `artifact/runs/full/e4_outrem_modext.csv`                                                                                                      |
| E5         | `artifact/runs/full/e5_textbadnets/onion.csv`, `artifact/runs/full/e5_textbadnets/dp.csv`                                                      |

Each row carries the cell it identifies (dataset, seed, and the swept knob)
followed by the metrics measured.
These are the columns to compare against the paper:

| Experiment | Metric columns                                                                                                                                                                                                              | Compare against                   |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| E1         | `robust_acc` (evasion), `pois_poison_acc` (poisoning), `fidelity` (model extraction), `online_auc` / `offline_auc` (membership inference), `attack_auc` (attribute inference), `ssim_avg` / `mse_avg` (data reconstruction) | Table 5                           |
| E2         | `defended_robust_acc`, `stolen_test_acc`, `fidelity`, `correct_fidelity`                                                                                                                                                    | Table 7                           |
| E3         | `acc_att_race`, `auc_race`, `acc_att_sex`, `auc_sex`, per `model_role` (baseline vs defended)                                                                                                                               | Table 6                           |
| E4         | `stolen_test_acc`, `fidelity`, `correct_fidelity`, per `percent`                                                                                                                                                            | Figures 3 and 4                   |
| E5         | `undef_asr` against `def_asr` (ONION) and `dp_asr` (DP-SGD), plus the `*_test_acc` columns                                                                                                                                  | Table 3 (ONION), Table 4 (DP-SGD) |

### Step 3: generate the tables and plots

Each paper table and plot has one script that renders it from the CSVs.
Rendering is a pure function of the CSV: no GPU, no model, no training.

```bash
uv run python artifact/make/make_all.py                              # all six, from runs/full
uv run python artifact/make/make_tab_advtr_modext.py                 # just E2's table
uv run python artifact/make/make_all.py --results-dir artifact/runs/smoke   # from another level
```

| Experiment | Script                                     | Output                                         | Compare against                     |
| ---------- | ------------------------------------------ | ---------------------------------------------- | ----------------------------------- |
| E1         | `artifact/make/make_tab_attack_results.py` | `tab_attack_results.tex`                       | Table 5                             |
| E2         | `artifact/make/make_tab_advtr_modext.py`   | `tab_advtr_modext.tex`                         | Table 7                             |
| E3         | `artifact/make/make_tab_attinf_advrtr.py`  | `tab_attinf_advrtr.tex`                        | Table 6                             |
| E4         | `artifact/make/make_fig_outrem.py`         | `fig_outrem_fid.png`, `fig_outrem_cor_fid.png` | Figures 3 and 4                     |
| E4         | `artifact/make/make_tab_outrem_modext.py`  | `tab_outrem_modext.tex`                        | (artifact-internal, no paper table) |
| E5         | `artifact/make/make_tab_textbadnets.py`    | `tab_textbadnets_interactions.tex`             | Table 3 (ONION), Table 4 (DP-SGD)   |

Tables land in `artifact/tables/generated/`, plots in
`artifact/plots/generated/` (as both `.png` and `.pdf`).
Every script reads whatever the CSVs hold, so a partial sweep still renders:
cells with no data come out blank, and each script prints a per-cell coverage
line naming exactly what is missing.

E5's single table holds both paper tables, one block each.
E4's table has no counterpart in the paper, which reports that study as Figures
3 and 4; it is a convenience view of the same CSV the figures use.

### Everything at once

```bash
artifact/run_all.sh     # every experiment at --level full, then render all six
```

This is Step 1 for all five experiments followed by Step 3, writing to
`artifact/runs/full/`.
`artifact/run_smoke.sh` is the same at `--level smoke`.

### Expected runtime

**L3 (`--level full`) is a single seed, seed 0 — not the paper's five-seed run.**
A reviewer runs each experiment once and checks whether their one number lands
within the paper's reported mean ± standard error; reproducing the exact means is
neither expected nor the point of the standard error. The five-seed column below
is the paper's own cost, shown only for context.

Costs are for a single NVIDIA A100, the reference host throughout (some E5
figures were originally measured on a slower A40; quoting them as A100 means a
real A100 run comes in at or under these numbers).

E5's row is measured, from the per-phase runtime columns in the paper's own
result CSVs.
The other four are estimates for one L3 seed: their `runtime_sec` column was
added after the paper run, so the first L3 run is what will replace these with
measurements.
The `--level smoke` sweep, by contrast, is already measured end to end (~12
minutes for all five experiments on one GPU: ~9 for E1-E4, ~2.5 for E5).
[`RUNTIME.md`](RUNTIME.md) holds the measured smoke breakdown, the per-phase
breakdown behind E5's number, and the method for regenerating the L3 figures.

| Experiment | L3: one seed (`--level full`) | Paper: 5 seeds        | What dominates                                                                                                                                                   |
| ---------- | ----------------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| E1         | 1 to 2 days (est)             | —                     | Membership inference trains a population of shadow models on CelebA; the other five attacks share one target each.                                               |
| E2         | 12 to 24 h (est)              | —                     | Adversarial training runs a PGD inner loop per batch (several times the cost of standard training), across 4 datasets x 4 budgets, plus a distillation per cell. |
| E3         | 4 to 8 h (est)                | —                     | Same adversarial training, but only 2 datasets (census, lfw) and no surrogate to distil.                                                                         |
| E4         | 12 to 24 h (est)              | —                     | kNN-Shapley scores the whole training set per cell, then the target is retrained and a surrogate distilled, across 4 datasets x 5 removal levels.                |
| E5         | 68 h ONION, 40 h DP-SGD       | ~540 h (~22 GPU-days) | Every cell fine-tunes LoRA adapters on a 3B-parameter Llama over SST-2 at a flat ~5 h; ONION adds 2.1 h scoring perplexity for the whole training corpus.        |

Every figure above is compute time on a host whose `artifact/setup_assets.py`
run has already completed; downloads are not included.

## Three verification levels

`--level` trades fidelity for time, so the pipeline can be checked before
committing to a full run.

| Level        | Question                    | Command                                                                                       | Cost                              |
| ------------ | --------------------------- | --------------------------------------------------------------------------------------------- | --------------------------------- |
| **L1 tests** | Does every script work?     | `uv run pytest artifact/tests` then `uv run pytest -m integration artifact/tests/integration` | seconds to minutes, CPU           |
| **L2 smoke** | Is the real pipeline sound? | `artifact/run_smoke.sh`                                                                       | ~12 min all five, one GPU         |
| **L3 full**  | Do we reproduce the paper?  | `artifact/run_all.sh`                                                                         | GPU-hours to GPU-days (see above) |

- **L1** runs the pure-logic unit tests (table formatting, spec hashing, CSV
  I/O, this doc's integrity) and a tiny end-to-end test per experiment on
  synthetic data. No GPU, no dataset download. This checks the code runs, not
  the paper's numbers.
- **L2** drives the same `run.py` scripts on real architectures (real VGG, real
  LoRA Llama), writing to `artifact/runs/smoke/`. It proves the pipeline end to
  end. Every repeated-work loop is shrunk to its floor: one epoch, a tenth of
  both the train and test splits, and reduced shadow-bank, inversion, and PGD
  counts (see [`RUNTIME.md`](RUNTIME.md)). Its numbers are therefore **not** the
  paper's; a reduced-budget run does not reproduce a trained model.
- **L3** is the reproduction, at the paper's per-experiment settings but for a
  **single seed (seed 0)**, writing to `artifact/runs/full/`. It is not the
  paper's five-seed sweep: a reviewer runs each experiment once and checks that
  their number falls within the paper's reported mean ± standard error. Only L3
  numbers are comparable with the paper at all. (`--seeds 0-4` reproduces the
  full five-seed means, but that is the authors' cost, not a reviewer's task.)

Each level writes to its own `artifact/runs/<level>/` tree, so a cheap check
never overwrites a full run's results.

## Claim to evidence

Each paper desideratum maps to a concrete command, test, or file in this tree.

| Desideratum          | Claim                                                                                                                    | Evidence                                                                                                                                                                                                                                                                                                                                              |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **D1 Comprehensive** | Eight risks, each with attacks, defenses, and metrics.                                                                   | The `amulet/` tree: `amulet/evasion/`, `amulet/poisoning/`, `amulet/unauth_model_ownership/`, `amulet/membership_inference/`, `amulet/attribute_inference/`, `amulet/distribution_inference/`, `amulet/data_reconstruction/`, `amulet/discriminatory_behavior/`. Module table in `AGENTS.md`.                                                         |
| **D2 Consistent**    | One uniform interface exposes every risk, so a defense for one risk and an attack for another compose into one pipeline. | [`CLAIMS.md`](CLAIMS.md) (Consistency), grounded in `examples/get_started.py` and `examples/attack_pipelines/`. `uv run pytest tests/test_api_conformance.py` enforces the shared defense entry point.                                                                                                                                                |
| **D3 Extensible**    | A new modality cost four new modules and one widened type, and reused DP-SGD unmodified.                                 | [`CLAIMS.md`](CLAIMS.md) (Extensibility), grounded in `examples/extending_amulet/custom_risk.md`, `examples/extending_amulet/custom_metric.md`, `examples/extending_amulet/custom_architecture.md`, and `examples/attack_pipelines/run_text_backdoor.py`. `uv run pytest tests/test_api_conformance.py` is the check that shaped ONION's entry point. |
| **D4 Applicable**    | Five experiments reproduce baseline attacks and three unintended interactions.                                           | The three steps above: run each experiment at `--level full`, then render with `uv run python artifact/make/make_all.py`, then compare against the paper's tables and figures.                                                                                                                                                                        |

## Layout

```text
artifact/
  ARTIFACT.md              # this file: start here
  CLAIMS.md                # design-claim demonstration (consistency, extensibility)
  RUNTIME.md               # per-phase runtime breakdown behind the estimates above
  common/                  # shared helpers: paths, config, models, io, registries
  experiments/             # one package per experiment; each has a run.py
    e1_attack_baselines/   #   baseline attack per risk
    e2_advtr_modext/       #   adversarial training x model ownership
    e3_advtr_attrinf/      #   adversarial training x attribute inference
    e4_outrem_modext/      #   outlier removal x model ownership
    e5_textbadnets/        #   text backdoor x ONION and x DP-SGD
  make/                    # one script per paper table/plot; make_all.py runs them
  runs/                    # run output, one subtree per level (gitignored)
  tables/generated/        # rendered .tex (gitignored)
  plots/generated/         # rendered .png and .pdf (gitignored)
  tests/                   # unit/ (pure logic) and integration/ (tiny end-to-end)
  setup_assets.py          # download every dataset and model weight; run this first
  run_experiments.py       # run every experiment at one level
  run_smoke.sh             # Level 2: all five at smoke, then render
  run_all.sh               # Level 3: all five at full, then render
```
