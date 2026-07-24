# Runtime

Where the time goes, per experiment and per phase. [`ARTIFACT.md`](ARTIFACT.md)
carries the headline per-experiment estimates; this file is the breakdown behind
them, the measured numbers where they exist, and the method for regenerating the
rest.

Every figure here is compute time. It assumes [`setup_assets.py`](setup_assets.py)
has already downloaded every dataset and model weight; downloads are not
included.

## Reference host

Every number below is quoted for a single **NVIDIA A100 (40 GB)**. The
`--level smoke` sweep was measured on one directly. E5's `--level full`
per-phase numbers were originally measured on a slower A40; quoting them as A100
is deliberate and conservative, because a real A100 run comes in at or under the
stated figure rather than over it.

Precision is fp32 throughout. DP-SGD's Opacus hooks do not compose with 4-bit or
fp16 layers, so E5 pins fp32 and the other experiments follow for comparability.
One experiment runs per GPU; nothing below assumes a second device. All numbers
are wall clock, not GPU-busy time, so they include data loading and evaluation as
well as training. A run on different hardware scales roughly with that GPU's fp32
throughput.

Note the two meanings of "full": `--level full` (L3) is the full paper setup for
a **single seed, seed 0** — what a reviewer runs and what the L3 figures describe.
The paper's five-seed sweep is a separate, larger cost, labelled as such
wherever it appears.

## How runtime is measured

Every result row records what it cost, so the tables below are recomputed from a
run rather than estimated by hand.

- **E1 through E4** write a `runtime_sec` column: wall clock from just after the
  cell's resume check to just before its row is written.
- **E5** predates that column and writes a finer breakdown instead, one column
  per training phase: `clean_train_runtime_sec`, `undef_train_runtime_sec`,
  `def_train_runtime_sec` and `onion_purify_runtime_sec` for ONION, and
  `clean_train_runtime_sec`, `undef_train_runtime_sec` and `dp_train_runtime_sec`
  for DP-SGD.

Read both with the model caches in mind, in two separate respects.

A baseline trained once and reused across cells is charged to whichever row
trained it, and repeated on the rows that reuse it. Summing a column down a CSV
therefore overstates the real cost; the sweep totals below de-duplicate the
shared phases first, and say so.

Two independent caches also decide whether a row's time reflects real work. The
CSV resume check skips **writing a row** whose cell is already recorded, while
the content-addressed checkpoint cache under `.model_cache/<level>/` skips
**training** a model it has already seen. They are not the same: a row absent
from the CSV is recomputed and written, but if its checkpoints are still on disk
the training is a load, and `runtime_sec` records seconds rather than the real
cost. A timing run must therefore start from a `.model_cache/<level>/` that does
not already hold the models it is about to train, or it silently understates. The
caches are per level, so a `smoke` timing is not corrupted by a previous `full`
run and vice versa, but re-timing the same level twice needs its cache cleared
between runs.

## `--level smoke` (measured)

Measured on the A100 above, from an empty `.model_cache/smoke/`, one seed. This
is what `run_smoke.sh` costs a reviewer once the assets are downloaded.

| Experiment |  Wall clock | Cells | What dominates                                                        |
| ---------- | ----------: | ----: | --------------------------------------------------------------------- |
| E1         |      5m 08s |    21 | Membership inference's 8-model shadow bank; evasion over 4 capacities |
| E2         |        49 s |    16 | PGD-7 adversarial training over 4 datasets x 4 budgets                |
| E3         |      1m 10s |    10 | PGD-7 adversarial training over 2 datasets x 5 budgets                |
| E4         |      1m 57s |    20 | kNN-Shapley over 4 datasets x 5 removal levels                        |
| E5         |      2m 35s |     2 | Two 1.1B-Llama fine-tunes per study over a 256-record slice           |
| **Total**  | **11m 39s** |    69 |                                                                       |

Smoke reduces every repeated-work loop, not just the data fraction (see
[`common/config.py`](common/config.py) and the `test_*_level_budget.py` tests).
The knobs that make these numbers what they are, all recorded in the CSVs:

| Knob                       |     full |          smoke |
| -------------------------- | -------: | -------------: |
| epochs                     |      100 |              1 |
| train / test fraction      |      1.0 |            0.1 |
| LiRA shadow models         |       64 |              8 |
| data-reconstruction alpha  |     3000 |             50 |
| PGD / evasion iterations   |       40 |              7 |
| E5 poison rates x epsilons |  5 / 5x2 |        1 / 1x1 |
| E5 target model            | 3B Llama | 1.1B TinyLlama |
| E5 train records           |      67k |            256 |

E5 is the exception that proves the rule about model size. E1-E4 keep their real
architectures at smoke because those are already cheap; E5's real architecture is
a 3B LLM, so smoke additionally swaps in a smaller real model (TinyLlama-1.1B)
and caps the corpus at a fixed 256 records rather than the level's 10% (which is
~6735 of SST-2's 67k). Every code path still runs; see `e5_textbadnets/onion.py`
`apply_level`.

### E1 per sub-attack (smoke)

Sum of `runtime_sec` over the cells the clean run wrote.

| Sub-attack           | Cells |    Sum | Note                                               |
| -------------------- | ----: | -----: | -------------------------------------------------- |
| membership_inference |     1 | 63.4 s | 8 shadow ResNets trained then scored               |
| evasion              |     4 | 98.3 s | one target per capacity, PGD-7 over the test split |
| data_reconstruction  |     4 | 43.2 s | 50 inversion steps per class                       |
| poisoning            |     4 | 36.7 s |                                                    |
| attribute_inference  |     4 | 34.9 s | shares its target with model extraction            |
| model_extraction     |     4 | 18.5 s | reuses the shared adversary-split target           |

### E4 per dataset (smoke)

kNN-Shapley is `O(train x test)` (see below), so the cost tracks each dataset's
test-split size. At smoke both splits are a tenth, which is what makes census
and the image sets cost seconds rather than the minutes they would at full.

| Dataset | Cells |    Sum | Per removal cell (percent > 0) |
| ------- | ----: | -----: | -----------------------------: |
| cifar   |     5 | 38.8 s |                         ~8.9 s |
| fmnist  |     5 | 38.0 s |                         ~9.2 s |
| census  |     5 | 27.8 s |                         ~6.6 s |
| lfw     |     5 |  1.8 s |                         ~0.2 s |

## E5 at `--level full` (measured)

Measured from the per-phase runtime columns in the paper's own result CSVs, on
SST-2 with a LoRA-adapted Llama-3.2-3B target. The paper's run covered 25 ONION
cells (5 seeds x 5 poison rates) and 30 DP-SGD cells (5 seeds x 3 poison rates x
2 privacy budgets); an L3 reviewer runs one seed of that, and the per-cell phase
costs below are what both are built from.

### Per-phase cost

Mean per cell, with the observed spread across cells.

| Phase                  | What it does                                                                        |  Mean | Range       |
| ---------------------- | ----------------------------------------------------------------------------------- | ----: | ----------- |
| `clean_train`          | Fine-tune the clean baseline target on unpoisoned SST-2                             | 5.2 h | 5.0 – 5.6 h |
| `undef_train`          | Fine-tune the undefended target on poisoned SST-2                                   | 5.1 h | 5.0 – 5.6 h |
| `def_train` (ONION)    | Fine-tune the target on ONION-purified poisoned data                                | 5.2 h | 5.2 – 5.6 h |
| `onion_purify` (ONION) | Perplexity-score and purify the whole training corpus plus the triggered test split | 2.1 h | 1.9 – 2.2 h |
| `dp_train` (DP-SGD)    | Train the target under Opacus per-sample clipping and noise                         | 1.8 h | 1.8 – 1.9 h |

LoRA fine-tuning at this scale is a flat ~5 h regardless of what the data has
been done to, so the poison rate and the privacy budget do not move the cost.
ONION's purification is not a preprocessing afterthought: at 2.1 h it is 40% of a
training run on its own, because it scores perplexity for every training
sentence. DP-SGD's own step is the cheapest phase here, because the paper's DP
schedule runs fewer epochs than the standard fine-tune it is compared against.

### L3 cost (one seed, what a reviewer runs)

Derived from the per-phase means and the caching the runners do. One L3 seed of
ONION is 5 cells (1 clean + 5 undefended + 5 defended + 5 purify) at ~68 h; one
L3 seed of DP-SGD is 6 cells at ~31 h.

| Study (one seed) | Phase accounting                               |                   Total |
| ---------------- | ---------------------------------------------- | ----------------------: |
| ONION            | 1 clean + 5 undefended + 5 defended + 5 purify |                   ~68 h |
| DP-SGD           | 1 clean + 3 undefended + 6 DP                  |                   ~31 h |
| **E5 L3 total**  |                                                | **~99 h (~4 GPU-days)** |

### Paper five-seed cost (for context, not an L3 task)

The paper's full sweep multiplies the per-seed cost across five independent
seeds. It is the authors' cost, shown so the per-seed number above has a scale;
a reviewer never runs it.

| Sweep           | Phase accounting                                  |                     Total |
| --------------- | ------------------------------------------------- | ------------------------: |
| ONION           | 5 clean + 25 undefended + 25 defended + 25 purify |                    ~340 h |
| DP-SGD          | 5 clean + 15 undefended + 30 DP                   |                    ~155 h |
| **Paper total** |                                                   | **~495 h (~21 GPU-days)** |

Seeds are independent and share no cache, so N nodes give close to an N-fold
speedup when the authors do run the full sweep.

**E5's `full` sweep does not need re-running.** The paper's result CSVs are the
ones this section is measured from. It is documented here because it is the one
experiment whose `full` cost is known rather than projected.

## E1 through E4 at `--level full`: not yet measured

These four have no measured `full` breakdown yet. Their `runtime_sec` column was
added after the paper run, so the first `--level full` sweep is what will
populate it. [`ARTIFACT.md`](ARTIFACT.md) carries order-of-magnitude estimates in
the meantime, and flags them as estimates.

Once a full run exists, regenerate this section from its CSVs:

```bash
uv run python - <<'PY'
import csv, glob, statistics
for path in sorted(glob.glob("artifact/runs/full/**/*.csv", recursive=True)):
    rows = list(csv.DictReader(open(path)))
    times = [float(row["runtime_sec"]) for row in rows if row.get("runtime_sec")]
    if times:
        print(f"{path:56} {len(times):3} cells  "
              f"mean {statistics.mean(times)/3600:5.2f} h  "
              f"total {sum(times)/3600:6.1f} h")
PY
```

For E1, group by `capacity` before averaging: the six attacks share one target
per (seed, capacity), so the attack that trains it absorbs the training cost and
the rest are much cheaper. For E2 and E4, group by `dataset`: the four differ
enough that a mean across them describes none of them. For E3, the baseline row
carries the shared clean-target training and each budget row carries only its own
work, so the rows are disjoint and can be summed directly.

### Why E4 will dominate at full, and why its cheap-looking datasets are not cheap

E4's per-cell cost is not training but `OutlierRemoval._knn_shapley`, a pure
Python double loop over train x test:

```python
for i in range(m):            # m = every test point
    for _ in range(n - 1):    # n = every train point
```

Measured at about 2.6 microseconds per inner iteration, so a cell costs roughly
`2.6e-6 * n * m` seconds before any retraining. Both factors matter, which has
two consequences at full that the smoke numbers above do not show. A small
**training** set does not imply a cheap cell, because `m` is the test split the
outer loop walks: census has the largest test split of the four (23,224 at full),
so despite being tabular it sits alongside the image datasets rather than below
them. And the gap between smoke and full is quadratic-ish, not linear: smoke cuts
**both** `n` and `m` to a tenth, so a full cell is roughly 100x its smoke cell,
not 10x.

| Dataset | n x m (full)    | Shapley (full, projected) |
| ------- | --------------- | ------------------------: |
| fmnist  | 30,000 x 10,000 |                    13 min |
| cifar   | 25,000 x 10,000 |                    11 min |
| census  | 11,611 x 23,224 |                    12 min |
| lfw     | 2,395 x 2,053   |                      13 s |

At `--level full` that is roughly 2.5 GPU-hours of pure Shapley across E4's 16
removal cells, on top of the retraining and distillation each cell also pays.
