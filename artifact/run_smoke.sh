#!/usr/bin/env bash
# Level 2 (smoke): run every experiment on real architectures for a bare-minimum
# budget (1 epoch, small data fraction), then regenerate every table and plot
# from the smoke results. Proves the real pipelines are sound without a full
# training budget. Output lands under artifact/runs/smoke/, never runs/full/.
#
# Smoke uses real models and, for E1-E4, real datasets on first use, so it wants
# a GPU and the dataset downloads; E5 additionally needs the `llm` extra.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== run_smoke.sh: running all experiments at smoke level =="
uv run python "${HERE}/run_experiments.py" --level smoke "$@"

echo "== run_smoke.sh: regenerating tables and plots from runs/smoke =="
uv run python "${HERE}/make/make_all.py" --results-dir "${HERE}/runs/smoke"
