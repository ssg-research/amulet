"""E1: one representative attack per risk, on CelebA.

Six sub-experiments, one per risk. Each sweeps the four VGG capacities `m1`-`m4`,
which the paper's table labels VGG11/13/16/19:

* `evasion` — `EvasionPGD` against a PGD-undefended target.
* `poisoning` — `BadNets`, comparing a clean and a backdoored target.
* `model_extraction` — `ModelExtraction` distilling a stolen surrogate.
* `membership_inference` — `LiRA` against an intentionally overfit ResNet, the
  one sub-experiment restricted to a single column.
* `attribute_inference` — `DudduCIKM2022` inferring CelebA's `Male` attribute.
* `data_reconstruction` — `FredriksonCCS2015` inverting the target per class.

The sub-experiments deliberately do **not** all share a target model: they
diverge in optimizer recipe, in which half of the training split the target saw,
and in which CelebA attribute is the label. Those divergences are encoded in the
`ModelSpec` fields, so sharing happens automatically where the recipes agree
(`model_extraction` and `attribute_inference`) and is impossible where they do
not. See `train_targets.py`, which builds every spec in one place.

Layout: `train_targets.py` defines every target (which model, how it trains),
`context.py` the run scaffolding (data loading, the checkpoint cache, level
scaling), and `attacks/` the six sub-attacks, each of which loads a target and
measures it. `run.py` is the uniform entry point the CLI, the level sweepers and
the tests use. Nothing is imported here: importing this package must not pull in
torch.
"""
