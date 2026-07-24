"""The artifact's experiment runners, one package per paper experiment.

`artifact/` is put on `sys.path` by each entry script rather than installed, so
these are imported by top-level name (`experiments.e5_textbadnets.run`), which
is what `common.registry` maps every experiment ID to.

Declaring this a regular package (not relying on a namespace package) makes the
artifact tree the unambiguous answer to `import experiments`, so it is never
shadowed by another `experiments/` directory earlier on `sys.path`.

Keep it free of imports: `common.registry` resolves experiment modules lazily so
that listing the experiments never pulls in torch.
"""
