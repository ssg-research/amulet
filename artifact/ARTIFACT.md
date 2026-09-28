# Scarab Benchmark Artifact

This directory reproduces the tables and figures in the Scarab paper.
Scarab (imported as `scarab`) is a PyTorch library for
evaluating unintended interactions among machine-learning defenses and risks
across security, privacy, and fairness.

No result data ships with this repository.
Every number comes from running an experiment here and comparing it against the
corresponding table or figure in the paper.

## Quickstart

To reproduce paper results, run the following commands. More details are below.

```bash
# 1. Install (CUDA 13 host; see Setup for other hardware).
uv sync --extra cu130 --extra llm --extra dev

# 2. Download the datasets and model weights every experiment reads.
uv run python artifact/setup_assets.py

# 3. Check the whole pipeline works, cheaply, on one GPU (minutes).
bash artifact/run_smoke.sh

# 4. Reproduce the paper (paper-scale training; see Expected runtime).
bash artifact/run_full.sh
```

Steps 3 and 4 each run all five experiments and then render every table and
figure. The rendered output lands in `artifact/tables/generated/` and
`artifact/plots/generated/`; compare it against the paper.

If you only want to confirm the code runs, without a GPU or any download, run the
tests instead (seconds to minutes, CPU):

```bash
uv run pytest artifact/tests                              # fast unit tests
uv run pytest -m integration artifact/tests/integration   # tiny end-to-end per experiment
```

## Setup

Dependencies are pinned and managed with `uv`.
The install needs one torch build extra matching your hardware, the `llm` extra
for the text experiment (E5), and `dev` for the tests.
The primary command above assumes a CUDA 13 host.
On other hardware, swap the torch extra:

```bash
uv sync --extra cu128 --extra llm --extra dev   # CUDA 12.x host
uv sync --extra cpu   --extra llm --extra dev   # no GPU (enough for the tests)
```

The `cpu`, `cu128`, and `cu130` extras are mutually exclusive; request exactly
one.
Check the maximum CUDA your driver supports with `nvidia-smi`.
Training at smoke or full scale needs a GPU; the `cpu` extra is enough for the
Quickstart tests only.

`setup_assets.py` fetches every dataset, preprocesses each one the experiments
read, and downloads the two pretrained models E5 fine-tunes (TinyLlama-1.1B for
smoke, Meta's Llama-3.2-3B for full); every other model is trained from scratch.
Run it once before any experiment:

```bash
uv run python artifact/setup_assets.py           # fetch everything this install can
uv run python artifact/setup_assets.py --list    # show the assets and their sizes
```

It downloads about 17 GB in total: about 1.9 GB of datasets (1.5 GB of it
CelebA), 2.2 GB for TinyLlama-1.1B, and 12.9 GB for Llama-3.2-3B. Without the
`llm` extra it downloads only the 1.9 GB of datasets.
On a 100 Mbit/s connection the transfers take about 25 minutes, and preprocessing
CelebA and LFW adds about 12 minutes on the CPU. CIFAR-10 comes from its original
host, which has served it at 70-150 kB/s in our runs, so that one 170 MB file
can add 20-40 minutes whatever the connection. Allow about an hour in all.

The runtimes quoted here and in [`RUNTIME.md`](RUNTIME.md) are compute time and
assume this download is already complete.
Downloading up front also keeps parallel runs from racing into the same `data/`
cache, and surfaces a gated repository or an expired token in minutes rather than
hours into a run.
E5's Llama weights are gated on the Hugging Face hub: accept the licence and run
`uv run hf auth login` before fetching them.
When the `llm` extra is absent the script skips E5's assets, so an E1-E4 reviewer
needs nothing more.

## Three verification levels

Every experiment runs at one of three levels, set with `--level`.
Each trades fidelity for time, so you can check the pipeline before committing to
a full run.

| Level     | Question                   | Command                      | Cost                    |
| --------- | -------------------------- | ---------------------------- | ----------------------- |
| **test**  | Does the code run?         | the Quickstart tests         | seconds to minutes, CPU |
| **smoke** | Is the pipeline sound?     | `bash artifact/run_smoke.sh` | minutes, one GPU        |
| **full**  | Do we reproduce the paper? | `bash artifact/run_full.sh`  | GPU-hours to GPU-days   |

- **test** substitutes tiny synthetic data and micro-architectures, so it runs
  anywhere with no download. It confirms the code runs; the numbers come from
  smoke and full.
- **smoke** drives the real `run.py` scripts on real architectures at a reduced
  budget: one epoch, a tenth of each split, and every repeated-work loop cut to
  its floor. It proves the pipeline end to end, at reduced-budget accuracy.
- **full** is the reproduction, at the paper's settings but for a single seed
  (seed 0). A reviewer runs each experiment once and checks that their number
  falls within the paper's reported mean ± standard error. Full is the level to
  compare against the paper.

Each level writes to its own `artifact/runs/<level>/` tree, so a cheap check
never overwrites a full run's results.

## Running one experiment

`run_smoke.sh` and `run_full.sh` run all five experiments. To run one on its own,
call its `run.py` with a `--level`:

```bash
uv run python artifact/experiments/e2_advtr_modext/run.py --level full
```

Every runner takes `--level {test,smoke,full}` and `--seeds` (`0`, or `0-4` for
the paper's five-seed means), plus the knobs its own sweep varies:

| Experiment | Runner                                            | Sweep knobs               | Studies                                    |
| ---------- | ------------------------------------------------- | ------------------------- | ------------------------------------------ |
| E1         | `artifact/experiments/e1_attack_baselines/run.py` | `--attacks --capacities`  | one baseline attack per risk               |
| E2         | `artifact/experiments/e2_advtr_modext/run.py`     | `--datasets --epsilons`   | adversarial training x model ownership     |
| E3         | `artifact/experiments/e3_advtr_attrinf/run.py`    | `--datasets --epsilons`   | adversarial training x attribute inference |
| E4         | `artifact/experiments/e4_outrem_modext/run.py`    | `--datasets --percents`   | outlier removal x model ownership          |
| E5         | `artifact/experiments/e5_textbadnets/run.py`      | `--which {onion,dp,both}` | text backdoor x ONION and x DP-SGD         |

```bash
# One dataset, one budget, one seed.
uv run python artifact/experiments/e2_advtr_modext/run.py --level full --datasets census --epsilons 0.01 --seeds 0
```

Runs resume: a sweep that dies partway is restarted with the same command, and
cells already written are skipped rather than recomputed.

## Reading the results

Each run appends one row per configuration to a CSV under
`artifact/runs/<level>/`. A full run writes to `artifact/runs/full/`:

| Experiment | CSV                                                                                                                         |
| ---------- | --------------------------------------------------------------------------------------------------------------------------- |
| E1         | `e1_attack_baselines/{evasion,poisoning,model_extraction,membership_inference,attribute_inference,data_reconstruction}.csv` |
| E2         | `e2_advtr_modext.csv`                                                                                                       |
| E3         | `e3_advtr_attrinf.csv`                                                                                                      |
| E4         | `e4_outrem_modext.csv`                                                                                                      |
| E5         | `e5_textbadnets/onion.csv`, `e5_textbadnets/dp.csv`                                                                         |

Each row opens with the cell it identifies (dataset, seed, and the swept knob)
followed by the metrics measured. These are the columns to compare against the
paper:

| Experiment | Metric columns                                                                                                                       | Compare against                   |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------- |
| E1         | `robust_acc`, `pois_poison_acc`, `fidelity`, `online_auc` / `offline_auc`, `attack_auc`, `ssim_avg` / `mse_avg` (one per sub-attack) | Table 5                           |
| E2         | `defended_robust_acc`, `stolen_test_acc`, `fidelity`, `correct_fidelity`                                                             | Table 7                           |
| E3         | `acc_att_race`, `auc_race`, `acc_att_sex`, `auc_sex`, per `model_role` (baseline vs defended)                                        | Table 6                           |
| E4         | `stolen_test_acc`, `fidelity`, `correct_fidelity`, per `percent`                                                                     | Table 8, Figures 3 and 4          |
| E5         | `undef_asr` against `def_asr` (ONION) and `dp_asr` (DP-SGD), plus the `*_test_acc` columns                                           | Table 3 (ONION), Table 4 (DP-SGD) |

## Rendering the tables and figures

Each paper table and figure has one renderer that reads the CSVs and writes the
output. Rendering is a pure function of the CSVs: no GPU, no model, seconds.
`run_smoke.sh` and `run_full.sh` call `make_all.py` for you; run it directly to
re-render without re-running the experiments:

```bash
uv run python artifact/make/make_all.py                              # all six, from runs/full
uv run python artifact/make/make_all.py --results-dir artifact/runs/smoke   # from another level
uv run python artifact/make/make_tab_advtr_modext.py                 # just E2's table
```

| Experiment | Renderer                                   | Output                                                     | Compare against                   |
| ---------- | ------------------------------------------ | ---------------------------------------------------------- | --------------------------------- |
| E1         | `artifact/make/make_tab_attack_results.py` | `tab_attack_results.tex`                                   | Table 5                           |
| E2         | `artifact/make/make_tab_advtr_modext.py`   | `tab_advtr_modext.tex`                                     | Table 7                           |
| E3         | `artifact/make/make_tab_advtr_attrinf.py`  | `tab_advtr_attrinf.tex`                                    | Table 6                           |
| E4         | `artifact/make/make_fig_outrem.py`         | `fig_outrem_fid.{png,pdf}`, `fig_outrem_cor_fid.{png,pdf}` | Figures 3 and 4                   |
| E4         | `artifact/make/make_tab_outrem_modext.py`  | `tab_outrem_modext.tex`                                    | Table 8                           |
| E5         | `artifact/make/make_tab_textbadnets.py`    | `tab_textbadnets_interactions.tex`                         | Table 3 (ONION), Table 4 (DP-SGD) |

Tables land in `artifact/tables/generated/`, figures in
`artifact/plots/generated/` (each as `.png` and `.pdf`).
Every renderer reads whatever the CSVs hold, so a partial run still renders:
missing cells come out blank, and each renderer prints a coverage line naming
exactly what is absent.
E5's one table holds both paper tables, one block each.
E4's Table 8 and its Figures 3 and 4 reproduce the same result from one CSV: the
table is the tabular form, the figures plot it.

## Expected runtime

Full is Level 3: it trains real models at paper scale.
Costs are for one full run on a single NVIDIA A100, the reference host throughout.
[`RUNTIME.md`](RUNTIME.md) holds the per-phase breakdown and the method for
regenerating it from a run's own `runtime_sec` columns.

All five full costs are measured: E5's from the paper's own result CSVs, E1
through E4 from a `--level full` run with one experiment per GPU.

| Experiment | Full (one run)                      | What dominates                                                                                                       |
| ---------- | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| E1         | ~21.7 h                             | training one target per capacity; membership inference's shadow-model bank is the largest single cost (~3.8 h)       |
| E2         | ~11.8 h                             | PGD adversarial training plus a distillation per cell; the cifar column alone is ~8.2 h                              |
| E3         | ~1.4 h                              | the same adversarial training, over 2 datasets and no surrogate                                                      |
| E4         | ~4.8 h                              | the retrain and distillation on the image datasets; kNN-Shapley's `train x test` loop on census                      |
| E5         | ~108 h (~68 h ONION + ~40 h DP-SGD) | every cell fine-tunes LoRA adapters on a 3B Llama at a flat ~5 h; ONION adds ~2 h scoring perplexity over the corpus |

E1 through E4 total about 40 h (~1.7 GPU-days); with E5, all five come to about
148 h (~6.2 GPU-days) on one A100. The experiments are independent, so running
each on its own GPU cuts the wall clock to the longest one, E5.

The smoke sweep is about 11 minutes for all five on one GPU;
[`RUNTIME.md`](RUNTIME.md) carries the measured per-experiment breakdown.

## Layout

```text
artifact/
  ARTIFACT.md              # this file: start here
  CLAIMS.md                # design-claim walkthrough (consistency, extensibility)
  RUNTIME.md               # per-phase runtime breakdown
  setup_assets.py          # step one: download every dataset and model weight
  run_experiments.py       # run every experiment at one level
  run_smoke.sh             # smoke: all five reduced, then render
  run_full.sh              # full: all five at paper scale, then render
  common/                  # shared infrastructure (paths, config, cache, io, training)
  experiments/             # one package per experiment
    e1_attack_baselines/   #   one baseline attack per risk
    e2_advtr_modext/       #   adversarial training x model ownership
    e3_advtr_attrinf/      #   adversarial training x attribute inference
    e4_outrem_modext/      #   outlier removal x model ownership
    e5_textbadnets/        #   text backdoor x ONION and x DP-SGD
  make/                    # one renderer per paper table/figure; make_all.py runs them
  tests/                   # unit/ (pure logic) and integration/ (tiny end-to-end)
```

Inside an experiment package, `train_targets.py` defines the models it trains,
`run.py` is the entry point, `schemas.py` fixes its CSV columns, and E1's six
attacks live in `attacks/`.

## Claim to evidence

Each paper desideratum maps to a concrete command, test, or file here.

| Desideratum          | Claim                                                                                    | Evidence                                                                                                                          |
| -------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **D1 Comprehensive** | Eight risks, each with attacks, defenses, and metrics.                                   | The `scarab/` tree, one package per risk; module table in [`AGENTS.md`](../AGENTS.md).                                            |
| **D2 Consistent**    | One uniform interface, so a defense for one risk composes with an attack for another.    | [`CLAIMS.md`](CLAIMS.md), grounded in `examples/`. `uv run pytest tests/test_api_conformance.py` enforces the shared entry point. |
| **D3 Extensible**    | A new modality (text) cost four modules and one widened type, reusing DP-SGD unmodified. | [`CLAIMS.md`](CLAIMS.md), grounded in `examples/extending_scarab/` and `examples/attack_pipelines/run_text_backdoor.py`.          |
| **D4 Applicable**    | Five experiments reproduce baseline attacks and three unintended interactions.           | The Quickstart: run each experiment at `--level full`, render, and compare against the paper.                                     |
