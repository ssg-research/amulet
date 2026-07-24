"""E1's six sub-attacks, one module per risk.

Each module exposes `run_cell(ctx, capacity, output_dir)`: it loads (or trains)
its target through `context.RunContext.get_or_train`, runs its attack, and
appends one result row. The target specs it trains live in the sibling
`train_targets` module; the run scaffolding in `context`; the generic training
recipes in `common.training`. Nothing is imported here: importing this package
must not pull in torch.
"""
