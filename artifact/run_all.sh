#!/usr/bin/env bash
# Level 3 (full): run every experiment at paper settings, then regenerate every
# table and plot from the full results. Output lands under artifact/runs/full/
# (never the committed results/). Authors promote a completed run by copying
# runs/full/<id>.csv into results/ and committing.
#
# WARNING: full is Level 3. It trains real models at paper scale and is
# GPU-HOURS to GPU-DAYS on a single consumer GPU (see RUNTIME.md), needs the
# dataset downloads, and E5 needs the `llm` extra. This is not a quick check;
# use run_smoke.sh to validate the pipeline first.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

cat <<'BANNER'
============================================================================
 run_all.sh: FULL (Level 3) run.
 This is paper-scale training: GPU-hours to GPU-days on one consumer GPU.
 Output goes to artifact/runs/full/ ; the committed results/ are left intact.
 Interrupt now (Ctrl-C) if you meant to run run_smoke.sh instead.
============================================================================
BANNER

echo "== run_all.sh: running all experiments at full level =="
uv run python "${HERE}/run_experiments.py" --level full "$@"

echo "== run_all.sh: regenerating tables and plots from runs/full =="
uv run python "${HERE}/make/make_all.py" --results-dir "${HERE}/runs/full"
