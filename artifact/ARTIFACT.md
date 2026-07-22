# Amulet Benchmark Artifact

This `artifact/` directory reproduces the tables and plots in the Amulet
benchmark submission.
Amulet (`amuletml` on PyPI, imported as `amulet`) is a Python library for
evaluating unintended interactions among machine-learning defenses and risks
across security, privacy, and fairness.

## What this is

The artifact **is the whole repository** at a pinned commit.
The library under review is in `amulet/`; the `artifact/` tree here holds the
scripts that reproduce the paper's results.
Read the library source directly, re-run the paper's experiments with the
`run.py` scripts here, and regenerate each table and plot with the wrappers.

Two kinds of claim back the paper, and both are reproducible from this tree:

- **Design claims** (consistent, extensible): the uniform interface and the
  cost of adding a modality. These are code and tests, demonstrated in
  [`CLAIMS.md`](CLAIMS.md) and enforced by `tests/test_api_conformance.py`.
- **Empirical claims** (applicable): five experiments backing five tables and
  two plots. Each paper table or plot has one wrapper that regenerates it.

What reproduces without a GPU:

- **E5 (text backdoor) reproduces the paper's numbers from committed data, with
  no GPU.** `artifact/results/e5_textbadnets/onion.csv` and
  `artifact/results/e5_textbadnets/dp.csv` ship in the repository, and the E5
  wrapper renders Tables 3 and 4 from them in seconds.
- **E1 through E4 are code-complete and reproduce each table's structure
  now.** They ship no result CSVs yet, so their wrappers render a correct blank
  skeleton and report per-cell coverage as MISSING. Their numbers arrive after a
  full (Level 3) run on a GPU.

## Setup

Dependencies are pinned and managed with `uv` (never `pip` or `conda`).
Pick one torch build extra matching the host, and add `dev` for the checks.
The text study (E5) additionally needs the `llm` extra.

```bash
uv sync --extra cpu --extra dev            # GPU-less / CI; enough for everything below except real GPU-scale runs
uv sync --extra cu128 --extra dev          # CUDA 12.x host
uv sync --extra cu130 --extra dev          # CUDA 13 host
uv sync --extra cu128 --extra dev --extra llm   # add the text/LLM stack for E5
```

Do not request more than one torch extra at once: `cpu`, `cu128`, and `cu130`
are declared conflicting and requesting several errors.
The `cpu` extra is sufficient for the whole zero-GPU path on this page: the
tests, the integration tier, and regenerating every table and plot from
committed data.
Real training at smoke or full scale requires a GPU.

## Three verification levels

The artifact answers three escalating questions.
Run them from the repository root.

| Level        | Question                    | Command                                                                                       | Cost                             |
| ------------ | --------------------------- | --------------------------------------------------------------------------------------------- | -------------------------------- |
| **L1 tests** | Does every script work?     | `uv run pytest artifact/tests` then `uv run pytest -m integration artifact/tests/integration` | seconds to minutes, CPU          |
| **L2 smoke** | Is the real pipeline sound? | `artifact/run_smoke.sh`                                                                       | minutes, one GPU (GPU-only)      |
| **L3 full**  | Do we reproduce the paper?  | `artifact/run_all.sh`                                                                         | GPU-hours to GPU-days (GPU-only) |

- **L1** runs the pure-logic unit tests (table formatting, spec hashing, CSV
  I/O, this doc's integrity) and a tiny end-to-end test per experiment on a
  handful of synthetic examples. It needs no GPU and no dataset download.
- **L2** drives the same `run.py` scripts as the paper on real architectures
  (real VGG, real LoRA Llama) for a one-epoch budget, then regenerates every
  table and plot from the smoke output under `artifact/runs/smoke/`. It trains
  real models, so it requires a GPU, downloads the vision datasets on first use,
  and needs the `llm` extra for E5.
- **L3** runs paper settings, one seed by default (`--seeds 0-4` for the paper's
  multi-seed means). It writes to `artifact/runs/full/`, leaving the committed
  `artifact/results/` intact; an author promotes a completed run by copying
  those CSVs into `artifact/results/`. Per-experiment wall-clock and hardware
  estimates will live in `RUNTIME.md`.

## Reproducing each table and plot

Each paper table or plot has one wrapper under `artifact/make/`.
A wrapper ensures results exist (or reads the committed CSV) and renders the
`.tex` or `.png`.
Rendering is a pure function of the CSV: no GPU, no model, no training.
Regenerate all six at once, reading the committed `artifact/results/`:

```bash
uv run python artifact/make/make_all.py
```

Generated files land in `artifact/tables/generated/` and
`artifact/plots/generated/`.
The paper's own tables and figures are not mirrored in this repository; compare
a generated table against the paper itself.
To render from your own re-run instead, pass
`--results-dir artifact/runs/full`.

| Experiment | Wrapper                                                  | Generates                                                   | Paper                             |
| ---------- | -------------------------------------------------------- | ----------------------------------------------------------- | --------------------------------- |
| E1         | `uv run python artifact/make/make_tab_attack_results.py` | `tab_attack_results.tex`                                    | Table 5                           |
| E2         | `uv run python artifact/make/make_tab_advtr_modext.py`   | `tab_advtr_modext.tex`                                      | Table 7                           |
| E3         | `uv run python artifact/make/make_tab_attinf_advrtr.py`  | `tab_attinf_advrtr.tex`                                     | Table 6                           |
| E4         | `uv run python artifact/make/make_fig_outrem.py`         | `fig_outrem_fid.png`, `fig_outrem_cor_fid.png`              | Figures 3 and 4                   |
| E4         | `uv run python artifact/make/make_tab_outrem_modext.py`  | `tab_outrem_modext.tex` (artifact-internal, no paper table) | --                                |
| E5         | `uv run python artifact/make/make_tab_textbadnets.py`    | `tab_textbadnets_interactions.tex`                          | Table 3 (ONION), Table 4 (DP-SGD) |

E4's table is an artifact convenience: the paper reports that study as Figures 3
and 4, so the figures are what an E4 run reproduces.
E5's single generated table holds both paper tables, one block each.
For E1 through E4 the wrappers print a per-cell coverage line showing which
cells a full run still needs to fill.

Run one experiment end-to-end at a chosen level with the registry-driven driver
(`run_experiments.py`) or the experiment's own `run.py`:

```bash
uv run python artifact/run_experiments.py --level test            # all five, tiny
uv run python artifact/run_experiments.py --level smoke --only e5_textbadnets   # one, real (GPU)
uv run python artifact/experiments/e5_textbadnets/run.py --level test           # E5 directly
```

`--level` selects the tier: `test` (L1), `smoke` (L2), or `full` (L3).
`--only` takes one experiment name:

| Experiment | `--only` name         | Runs                                       |
| ---------- | --------------------- | ------------------------------------------ |
| E1         | `e1_attack_baselines` | baseline attack per risk                   |
| E2         | `e2_advtr_modext`     | adversarial training x model ownership     |
| E3         | `e3_advtr_attrinf`    | adversarial training x attribute inference |
| E4         | `e4_outrem_modext`    | outlier removal x model ownership          |
| E5         | `e5_textbadnets`      | text backdoor x ONION and x DP-SGD         |

## Reading a run's output

Each `run.py` appends one row per configuration to a CSV under `runs/<level>/`,
which mirrors the `results/` layout; a run never writes into the committed
`results/`.
After a run, open its CSV and compare the metric columns below against the
corresponding table or figure in the paper.

| Experiment | Output CSV (under `runs/<level>/`)                  | Metric columns to compare                                                                                                                                                                                                   | Compare against (paper)           |
| ---------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| E1         | `e1_attack_baselines/<risk>.csv` (one CSV per risk) | `robust_acc` (evasion), `pois_poison_acc` (poisoning), `attack_auc` (attribute inference), `online_auc` / `offline_auc` (membership inference), `fidelity` (model extraction), `ssim_avg` / `mse_avg` (data reconstruction) | Table 5                           |
| E2         | `e2_advtr_modext.csv`                               | `defended_robust_acc`, `stolen_test_acc`, `fidelity`, `correct_fidelity`                                                                                                                                                    | Table 7                           |
| E3         | `e3_advtr_attrinf.csv`                              | `acc_att_race`, `auc_race`, `acc_att_sex`, `auc_sex` (per `model_role`: baseline vs defended)                                                                                                                               | Table 6                           |
| E4         | `e4_outrem_modext.csv`                              | `stolen_test_acc`, `fidelity`, `correct_fidelity` (per `percent`)                                                                                                                                                           | Figures 3 and 4                   |
| E5         | `e5_textbadnets/onion.csv`, `e5_textbadnets/dp.csv` | `undef_asr` against `def_asr` (ONION) and `dp_asr` (DP-SGD); the `*_test_acc` columns                                                                                                                                       | Table 3 (ONION), Table 4 (DP-SGD) |

For example, a full E2 run:

```bash
uv run python artifact/experiments/e2_advtr_modext/run.py --level full
```

writes `artifact/runs/full/e2_advtr_modext.csv`, one row per `(dataset, seed, epsilon)` cell.
Read `defended_robust_acc` and the surrogate's `fidelity`, then compare them
against the matching cells of Table 7 in the paper.

E5 is the one experiment whose CSVs already ship in `results/`
(`artifact/results/e5_textbadnets/onion.csv` and `dp.csv`), so its numbers can be
read and compared without a run.

## Claim to evidence

Each paper desideratum maps to a concrete command, test, or file in this tree.

| Desideratum          | Claim                                                                                                                    | Evidence                                                                                                                                                                                                                                                                                                                                              |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **D1 Comprehensive** | Eight risks, each with attacks, defenses, and metrics.                                                                   | The `amulet/` tree: `amulet/evasion/`, `amulet/poisoning/`, `amulet/unauth_model_ownership/`, `amulet/membership_inference/`, `amulet/attribute_inference/`, `amulet/distribution_inference/`, `amulet/data_reconstruction/`, `amulet/discriminatory_behavior/`. Module table in `AGENTS.md`.                                                         |
| **D2 Consistent**    | One uniform interface exposes every risk, so a defense for one risk and an attack for another compose into one pipeline. | [`CLAIMS.md`](CLAIMS.md) (Consistency), grounded in `examples/get_started.py` and `examples/attack_pipelines/`. `uv run pytest tests/test_api_conformance.py` enforces the shared defense entry point.                                                                                                                                                |
| **D3 Extensible**    | A new modality cost four new modules and one widened type, and reused DP-SGD unmodified.                                 | [`CLAIMS.md`](CLAIMS.md) (Extensibility), grounded in `examples/extending_amulet/custom_risk.md`, `examples/extending_amulet/custom_metric.md`, `examples/extending_amulet/custom_architecture.md`, and `examples/attack_pipelines/run_text_backdoor.py`. `uv run pytest tests/test_api_conformance.py` is the check that shaped ONION's entry point. |
| **D4 Applicable**    | Five experiments reproduce baseline attacks and three unintended interactions.                                           | The five wrappers above via `uv run python artifact/make/make_all.py`. E5 reproduces numerically from committed data; E1 through E4 reproduce structure now and numbers after L3.                                                                                                                                                                     |

## Layout

```text
artifact/
  ARTIFACT.md              # this file: start here
  CLAIMS.md                # design-claim demonstration (consistency, extensibility)
  common/                  # shared helpers: paths, config, models, io, registries
  experiments/             # one package per experiment
    e1_attack_baselines/   #   baseline attack per risk
    e2_advtr_modext/       #   adversarial training x model ownership
    e3_advtr_attrinf/      #   adversarial training x attribute inference
    e4_outrem_modext/      #   outlier removal x model ownership
    e5_textbadnets/        #   text backdoor x ONION and x DP-SGD
  make/                    # one wrapper per paper table/plot; make_all.py runs them
  tables/generated/        # regenerated .tex (gitignored)
  plots/generated/         # regenerated .png (gitignored)
  results/                 # committed ground-truth CSVs (E5 ships data)
  runs/                    # reviewer re-runs and author sweeps (gitignored)
  tests/                   # unit/ (pure logic) and integration/ (tiny end-to-end)
  run_experiments.py       # run every experiment at one level
  run_smoke.sh             # Level 2: all five at smoke, then regenerate
  run_all.sh               # Level 3: all five at full, then regenerate
```
