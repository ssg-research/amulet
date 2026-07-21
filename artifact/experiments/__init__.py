"""The artifact's experiment runners, one package per paper experiment.

`artifact/` is put on `sys.path` by each entry script rather than installed, so
these are imported by top-level name (`experiments.e5_textbadnets.run`), which
is what `common.registry` maps every experiment ID to.

This file is not decoration. The repository root carries its own `experiments/`
directory holding the live cluster sweep, and without a regular package here
that directory can answer an `import experiments` first (a namespace portion is
only a fallback, but it wins when nothing better follows). Declaring this one a
regular package makes the artifact tree the unambiguous answer.

Keep it free of imports: `common.registry` resolves experiment modules lazily so
that listing the experiments never pulls in torch.
"""
