#!/usr/bin/env bash
# Level 2 (smoke): run every experiment on real architectures for a bare-minimum
# budget (1 epoch, a tenth of both splits, every repeated-work loop shrunk to its
# floor), then regenerate every table and plot from the smoke results. Proves the
# real pipelines are sound without a full training budget: ~12 minutes for all
# five on one GPU (~9 for E1-E4, ~2.5 for E5). Output lands under
# artifact/runs/smoke/, never runs/full/.
#
# Smoke uses real models and datasets, so it wants a GPU; E5 additionally needs
# the `llm` extra. Run setup_assets.py first to fetch every dataset and model.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== run_smoke.sh: running all experiments at smoke level =="
uv run python "${HERE}/run_experiments.py" --level smoke "$@"

echo "== run_smoke.sh: regenerating tables and plots from runs/smoke =="
uv run python "${HERE}/make/make_all.py" --results-dir "${HERE}/runs/smoke"
