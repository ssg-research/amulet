# Runtime

Where the time goes, per experiment and per phase.
[`ARTIFACT.md`](ARTIFACT.md) carries the headline numbers; this file is the
breakdown behind them and the method for regenerating it.

Every figure here is compute time.
It assumes [`setup_assets.py`](setup_assets.py) has already downloaded every
dataset and model weight; downloads are not included.

## Reference host

Every number is quoted for a single **NVIDIA A100 (40 GB)**.
Precision is fp32 throughout: DP-SGD's Opacus hooks do not compose with 4-bit or
fp16 layers, so E5 pins fp32 and the others follow for comparability.
One experiment runs per GPU.
All numbers are wall clock, so they include data loading and evaluation, not just
training. A different GPU scales roughly with its fp32 throughput.

"Full" here means `--level full` for a single seed (seed 0), which is what a
reviewer runs. The paper's five-seed sweep is a separate, larger cost, labelled
as such wherever it appears.

## How runtime is measured

Every result row records what it cost, so these tables are recomputed from a run
rather than estimated by hand.

- **E1 through E4** write a `runtime_sec` column: wall clock from just after the
  cell's resume check to just before its row is written.
- **E5** writes a finer breakdown, one column per training phase
  (`clean_train_runtime_sec`, `undef_train_runtime_sec`, `def_train_runtime_sec`,
  and `onion_purify_runtime_sec` for ONION; the first three plus `dp_train_runtime_sec`
  for DP-SGD).

Read these with the model cache in mind, in two respects.

A baseline trained once and reused across cells is charged to whichever row
trained it, then repeated on the rows that reuse it. Summing a column down a CSV
therefore overstates the real cost; the totals below de-duplicate the shared
phases first.

Two independent caches also decide whether a row's time reflects real work. The
CSV resume check skips **writing a row** whose cell is already recorded; the
content-addressed checkpoint cache under `.model_cache/<level>/` skips **training**
a model it already holds. A row absent from the CSV is recomputed and written,
but if its checkpoint is still on disk the training is only a load, and
`runtime_sec` records seconds rather than the real cost. A timing run must
therefore start from a `.model_cache/<level>/` that does not already hold the
models it is about to train. The caches are per level, so a `smoke` timing is not
corrupted by a previous `full` run.

## `--level smoke`

This is what `run_smoke.sh` costs once the assets are downloaded: all five
experiments at a reduced budget, one seed. Measured on the reference host from a
cold `.model_cache/smoke/`.

| Experiment |  Wall clock | What dominates                                                   |
| ---------- | ----------: | ---------------------------------------------------------------- |
| E1         |      5m 26s | membership inference's shadow bank; evasion over four capacities |
| E2         |         39s | PGD adversarial training over 4 datasets x 4 budgets             |
| E3         |         48s | PGD adversarial training over 2 datasets x budgets               |
| E4         |      1m 46s | kNN-Shapley over 4 datasets x 5 removal levels                   |
| E5         |      2m 23s | two 1.1B-Llama fine-tunes per study over a small fixed slice     |
| **Total**  | **11m 02s** |                                                                  |

Rendering the tables and figures adds about 15 seconds, so `run_smoke.sh` end to
end is about 11.5 minutes.

Smoke reduces every repeated-work loop, not just the data fraction (see
[`common/config.py`](common/config.py) and the `test_*_level_budget.py` tests).
The knobs that set these numbers, all recorded in the CSVs:

| Knob                       |     full |          smoke |
| -------------------------- | -------: | -------------: |
| epochs                     |      100 |              1 |
| train / test fraction      |      1.0 |            0.1 |
| LiRA shadow models         |       64 |              8 |
| data-reconstruction alpha  |     3000 |             50 |
| PGD / evasion iterations   |       40 |              7 |
| E5 poison rates x epsilons |  5 / 4x2 |        1 / 1x1 |
| E5 target model            | 3B Llama | 1.1B TinyLlama |
| E5 train records           |      67k |            256 |

E1-E4 keep their real architectures at smoke because those are already cheap.
E5's real architecture is a 3B LLM, so smoke additionally swaps in a smaller real
model (TinyLlama-1.1B) and caps the corpus at a fixed 256 records.
Every code path still runs; see `apply_level` in `e5_textbadnets/onion.py`.

## `--level full`

### E5 (measured)

Measured from the per-phase runtime columns in the paper's own result CSVs, on
SST-2 with a LoRA-adapted Llama-3.2-3B target.
E5 is the one experiment whose full cost is known rather than projected, because
the paper's run wrote it.
(Some cells were originally measured on a slower A40; quoting them as A100 is
conservative, since a real A100 run comes in at or under these figures.)

Per cell, LoRA fine-tuning at this scale is a flat ~5 h regardless of what the
data has had done to it, so the poison rate and privacy budget do not move the
cost. ONION's purification is not a preprocessing afterthought: at ~2.1 h it is
40% of a training run on its own, because it scores perplexity for every training
sentence. DP-SGD's own step is the cheapest phase, because the paper's DP
schedule runs fewer epochs than the fine-tune it is compared against.

| Phase                  | What it does                                              |   Mean |
| ---------------------- | --------------------------------------------------------- | -----: |
| `clean_train`          | fine-tune the clean baseline on unpoisoned SST-2          | ~5.2 h |
| `undef_train`          | fine-tune the undefended target on poisoned SST-2         | ~5.1 h |
| `def_train` (ONION)    | fine-tune on ONION-purified poisoned data                 | ~5.2 h |
| `onion_purify` (ONION) | perplexity-score and purify the corpus and triggered test | ~2.1 h |
| `dp_train` (DP-SGD)    | train under Opacus per-sample clipping and noise          | ~1.8 h |

One full seed of ONION is ~68 h (1 clean + 5 undefended + 5 defended + 5
purify); one full seed of DP-SGD is ~40 h (1 clean + 4 undefended + 8 DP). E5's
full cost for one seed is therefore **~108 h (~4.5 GPU-days)**.
The paper's five-seed sweep is ~540 h (~22 GPU-days); seeds are independent and
share no cache, so N nodes give close to an N-fold speedup.
E5's full sweep does not need re-running: the paper's CSVs are what this section
is measured from.

### E1 through E4 (not yet measured)

These four have no measured full breakdown yet.
Their `runtime_sec` column was added after the paper run, so the first
`--level full` sweep is what will populate this section.

Once a full run exists, regenerate this from its CSVs:

```bash
uv run python - <<'PY'
import csv, glob, statistics
for path in sorted(glob.glob("artifact/runs/full/**/*.csv", recursive=True)):
    rows = list(csv.DictReader(open(path)))
    times = [float(row["runtime_sec"]) for row in rows if row.get("runtime_sec")]
    if times:
        print(f"{path:56} {len(times):3} cells  "
              f"mean {statistics.mean(times) / 3600:5.2f} h  "
              f"total {sum(times) / 3600:6.1f} h")
PY
```

Group before averaging, because a mean over unlike cells describes none of them:
E1 by `capacity` (the six attacks share one target per seed and capacity, so the
attack that trains it absorbs the cost), E2 and E4 by `dataset`. E3's baseline
row carries the shared clean-target training and each budget row only its own
work, so E3's rows are disjoint and sum directly.

E4 is expected to dominate, for a reason the smoke numbers hide. Its per-cell
cost is not training but `OutlierRemoval._knn_shapley`, a double loop over
`train x test`. Both factors matter: a small training set does not imply a cheap
cell, because the outer loop walks the **test** split, and census has the largest
test split of the four despite being tabular. And the gap from smoke to full is
quadratic, not linear: smoke cuts both factors to a tenth, so a full cell is
roughly 100x its smoke cell.

| Dataset | train x test (full) |
| ------- | ------------------- |
| fmnist  | 30,000 x 10,000     |
| cifar   | 25,000 x 10,000     |
| census  | 11,611 x 23,224     |
| lfw     | 2,395 x 2,053       |
