# Amulet Benchmark Artifact

This is the reproduction harness for the Amulet benchmark submission.
Amulet (`amuletml` on PyPI, imported as `amulet`) is a PyTorch library for
evaluating unintended interactions among machine-learning defenses and risks
across security, privacy, and fairness.
A reviewer opens this file first.

## What this is

The artifact **is the whole repository** at a pinned commit, not a packaged
black box.
The library under review lives in `amulet/`, and this `artifact/` tree is the
reproduction harness laid over it.
A reviewer reads the library source directly and re-runs the paper's
experiments against it with the wrappers here.

Two kinds of claim back the paper, and both are reproducible from this tree:

- **Design claims** (consistent, extensible): the uniform interface and the
  cost of adding a modality. These are code and tests, demonstrated in
  [`CLAIMS.md`](CLAIMS.md) and enforced by `tests/test_api_conformance.py`.
- **Empirical claims** (applicable): five experiments backing five tables and
  two plots. Each has one wrapper that regenerates its paper artifact.

Reproduction is honest about what runs without a GPU:

- **E5 (text backdoor) reproduces the paper table numerically from committed
  data, with no GPU.** `artifact/results/e5_textbadnets/onion.csv` and
  `artifact/results/e5_textbadnets/dp.csv` ship in the repository, and the E5
  wrapper renders Table 5 from them in seconds.
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
Real training at smoke or full scale wants a GPU.

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
  real models, so it wants a GPU, downloads the vision datasets on first use,
  and needs the `llm` extra for E5. **Do not expect this to be quick.**
- **L3** runs paper settings, one seed by default (`--seeds 0-4` for the paper's
  multi-seed means). It writes to `artifact/runs/full/`, leaving the committed
  `artifact/results/` intact; an author promotes a completed run by copying
  those CSVs into `artifact/results/`. Per-experiment wall-clock and hardware
  estimates will live in `RUNTIME.md`.

## Reproducing each table and plot

Each paper artifact has one wrapper under `artifact/make/`.
A wrapper ensures results exist (or reads the committed CSV) and renders the
`.tex` or `.png`.
Rendering is a pure function of the CSV: no GPU, no model, no training.
Regenerate all six at once, reading the committed `artifact/results/`:

```bash
uv run python artifact/make/make_all.py
```

Generated files land in `artifact/tables/generated/` and
`artifact/plots/generated/`; the hand-authored references stay in
`artifact/tables/` and `artifact/plots/`.
To render from a reviewer's own re-run instead, pass
`--results-dir artifact/runs/full`.

| Experiment | Paper artifact                         | Reference                                                                    | Wrapper                                                  | Reproduces                                   |
| ---------- | -------------------------------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------- | -------------------------------------------- |
| E1         | `tab_attack_results`                   | `artifact/tables/tab_attack_results.tex`                                     | `uv run python artifact/make/make_tab_attack_results.py` | structure now; numbers after L3              |
| E2         | `tab_advtr_modext`                     | `artifact/tables/tab_advtr_modext.tex`                                       | `uv run python artifact/make/make_tab_advtr_modext.py`   | structure now; numbers after L3              |
| E3         | `tab_attinf_advrtr`                    | `artifact/tables/tab_attinf_advrtr.tex`                                      | `uv run python artifact/make/make_tab_attinf_advrtr.py`  | structure now; numbers after L3              |
| E4         | `tab_outrem_modext`                    | `artifact/tables/tab_outrem_modext.tex`                                      | `uv run python artifact/make/make_tab_outrem_modext.py`  | structure now; numbers after L3              |
| E4         | `fig_outrem_fid`, `fig_outrem_cor_fid` | `artifact/plots/fig_outrem_fid.png`, `artifact/plots/fig_outrem_cor_fid.png` | `uv run python artifact/make/make_fig_outrem.py`         | structure now; numbers after L3              |
| E5         | `tab_textbadnets_interactions`         | `artifact/tables/tab_textbadnets_interactions.tex`                           | `uv run python artifact/make/make_tab_textbadnets.py`    | **numerically, from committed data, no GPU** |

The tables compare numeric cells only; a reworded caption never fails a check
and a changed number always does.
E5's regenerated table matches the reference byte-for-byte.
For E1 through E4 the wrappers print a per-cell coverage line so a reviewer sees
exactly which numbers a run still owes.

Run one experiment end-to-end at a chosen level with the registry-driven driver
or an experiment's own runner:

```bash
uv run python artifact/run_experiments.py --level test            # all five, tiny
uv run python artifact/run_experiments.py --level smoke --only e5_textbadnets   # one, real (GPU)
uv run python artifact/experiments/e5_textbadnets/run.py --level test           # E5 directly
```

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
  ARTIFACT.md              # this file: the reviewer's entry point
  CLAIMS.md                # design-claim demonstration (consistency, extensibility)
  common/                  # shared harness: paths, config, models, io, registries
  experiments/             # one package per experiment
    e1_attack_baselines/   #   baseline attack per risk
    e2_advtr_modext/       #   adversarial training x model ownership
    e3_advtr_attrinf/      #   adversarial training x attribute inference
    e4_outrem_modext/      #   outlier removal x model ownership
    e5_textbadnets/        #   text backdoor x ONION and x DP-SGD
  make/                    # one wrapper per paper table/plot; make_all.py runs them
  tables/                  # reference .tex; generated/ holds regenerated output
  plots/                   # reference .png; generated/ holds regenerated output
  results/                 # committed ground-truth CSVs (E5 ships data)
  runs/                    # reviewer re-runs and author sweeps (gitignored)
  tests/                   # unit/ (pure logic) and integration/ (tiny end-to-end)
  run_experiments.py       # run every experiment at one level
  run_smoke.sh             # Level 2: all five at smoke, then regenerate
  run_all.sh               # Level 3: all five at full, then regenerate
```
