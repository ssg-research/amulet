"""E5: the textual backdoor and its two defenses on SST-2 (plan §3, §12 P1).

Adopted from the live sweep in the repository-root `experiments/` tree, which
stays where it is: this is a copy wearing the artifact harness (`common.io`
result CSVs, `common.config` levels, the registry entry point) over the same
LLM internals.

Two sub-experiments share one victim recipe, one dataset loader and one set of
caches:

* `onion` — ONION (a poisoning defense) against the poisoning attack, the
  *intended* interaction, swept over five poison rates.
* `dp` — DP-SGD (a membership-inference defense) against the same attack, the
  *unintended* interaction, swept over poison rate x target epsilon.

`run.py` is the uniform entry point both the CLI and the tests use. Nothing is
imported here: importing this package must not pull in torch or Hugging Face.
"""
