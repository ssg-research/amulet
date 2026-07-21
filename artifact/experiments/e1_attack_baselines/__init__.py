"""E1: one representative attack per risk, on CelebA (plan §3, §12 P2).

Six sub-experiments, one per risk, ported from the pre-modernisation scripts in
the old repository onto the artifact harness (`common.models` for the shared
checkpoint cache, `common.io` for the result CSVs, `common.config` for the
verification levels). Each sweeps the four VGG capacities `m1`-`m4`, which the
paper's table labels VGG11/13/16/19:

* `evasion` — `EvasionPGD` against a PGD-undefended target.
* `poisoning` — `BadNets`, comparing a clean and a backdoored victim.
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
not. See `shared.py`, which builds every spec in one place.

`run.py` is the uniform entry point the CLI, the level sweepers and the tests
use. Nothing is imported here: importing this package must not pull in torch.
"""
