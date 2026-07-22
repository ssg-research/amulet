# Design Claims: Consistency and Extensibility

This document substantiates the two design desiderata the paper argues by
construction: **D2 (consistent)** from the framework section, and **D3
(extensible)** from the extensibility section.
It is a companion to [`ARTIFACT.md`](ARTIFACT.md), which covers the empirical
claims.

Every snippet below is lifted from a runnable example in `examples/` and uses
the real API.
The single executable proof that ties both claims together is the conformance
test:

```bash
uv run pytest tests/test_api_conformance.py
```

## Consistency (D2)

Each risk in Amulet follows one interface.
An attack or defense takes a PyTorch `nn.Module` and hyperparameters (poisoning
attacks take a dataset instead), and its output feeds the matching metric.
Modules sit at a predictable path: `amulet.<risk>.attacks.<attack>`,
`amulet.<risk>.defenses.<defense>`, and `amulet.<risk>.metrics.<metric>`.

### Every algorithm is an object you instantiate, then a call you make

Swapping one risk's attack for another changes the import and the constructor,
never the shape of the code.
Both of these come from `examples/get_started.py`:

```python
from amulet.evasion.attacks import EvasionPGD

# Instantiate the attack around the model, then run it.
evasion = EvasionPGD(target_model, test_loader, device, batch_size=256, epsilon=0.1)
adv_loader = evasion.attack()
```

A defense for the same risk exposes its risk's training entry point,
`train_robust`, and returns a defended `nn.Module`:

```python
from amulet.evasion.defenses import AdversarialTrainingPGD

adv_training = AdversarialTrainingPGD(
    target_model, criterion, optimizer, train_loader, device, epochs, epsilon=0.1
)
defended_model = adv_training.train_robust()
```

The training entry point is fixed per risk, not per defense: poisoning and
evasion defenses expose `train_robust`, the membership-inference defense exposes
`train_private`, fairness exposes `train_fair`, and the ownership defenses
expose `watermark` or `fingerprint`.
`tests/test_api_conformance.py` enforces this mechanically across every defense
in the package.

### A defense for one risk composes with an attack for another

Because the interface is uniform, a defense built for one risk and an attack
built for another compose into a single measurable pipeline.
This is exactly what experiment E2 does: it adversarially trains a model (a
defense for **evasion**), then extracts a surrogate from it (an attack for
**unauthorized model ownership**), and scores the surrogate with the ownership
risk's own metric.
The pattern is drawn from `examples/defense_pipelines/run_adversarial_training.py`
and `examples/attack_pipelines/run_model_extraction.py`, and lives in
`artifact/experiments/e2_advtr_modext/run.py`:

```python
from amulet.evasion.defenses import AdversarialTrainingPGD
from amulet.unauth_model_ownership.attacks import ModelExtraction
from amulet.unauth_model_ownership.metrics import evaluate_extraction

# Defense from the evasion risk.
defended = AdversarialTrainingPGD(
    model, criterion, optimizer, train_loader, device, epochs, epsilon=0.1
).train_robust()

# Attack from the model-ownership risk, against that defended model.
stolen = ModelExtraction(
    defended, surrogate, optimizer, adv_train_loader, device, epochs
).attack()

# Metric from the model-ownership risk consumes both models.
results = evaluate_extraction(defended, stolen, test_loader, device)
```

No adapter sits between the evasion defense and the ownership attack.
The defended model is an `nn.Module`, which is exactly what `ModelExtraction`
expects.
Experiment E4 composes the same ownership attack with a **poisoning** defense
(`OutlierRemoval`) the same way.

## Extensibility (D3)

The paper extends Amulet to text, a modality it did not previously support, and
measures what that cost.
The answer is four new modules and one widened type, with every existing risk,
attack, defense, and metric left intact.

### Four new modules, one per real import path

All four are exercised together in
`examples/attack_pipelines/run_text_backdoor.py`, which is also experiment E5.

| Module              | Import path                            | Role                                                                                                                                     |
| ------------------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `TextTensorDataset` | `amulet.datasets.TextTensorDataset`    | A tokenized text dataset, subclassing PyTorch's `TensorDataset` so existing consumers receive the type they already expect.              |
| `HFCausalLM`        | `amulet.models.HFCausalLM`             | A pretrained causal LM with LoRA adapters and a linear classification head, subclassing `AmuletModel` (`amulet/models/hf_causal_lm.py`). |
| `TextBadNets`       | `amulet.poisoning.attacks.TextBadNets` | The textual BadNets attack: insert a trigger word and relabel to the target.                                                             |
| `ONION`             | `amulet.poisoning.defenses.ONION`      | A perplexity-based input-purification defense against poisoning.                                                                         |

Each implements the abstract base class for its risk, so the metrics already
implemented for poisoning consume their outputs without modification:

```python
from amulet.datasets import TextTensorDataset, load_sst2
from amulet.models import HFCausalLM
from amulet.poisoning.attacks import TextBadNets

data = load_sst2(path=data_path, tokenizer_name=model_name, max_length=128)
attack = TextBadNets(trigger="cf", trigger_label=1, portion=0.1, random_seed=0)
poisoned_train = attack.poison_train(data.train_set)   # same return type as the image BadNets
poisoned_test = attack.poison_test(data.test_set)
```

### One widened type

Beyond the package export lists, the extension modified one existing file.
It added the text dataset class and widened the `AmuletDataset.modality` field
to admit text alongside images and tabular records
(`amulet/datasets/__data.py`):

```python
modality: Literal["image", "tabular", "text"]
```

No existing risk, attack, defense, or metric changed to make text work.

### A privacy defense, reused unmodified, measures a poisoning attack on an LLM

The second study added no module at all.
Amulet already implements DP-SGD as a defense against membership inference, and
it applies to the LoRA-adapted victim untouched.
A privacy defense written and tested against vision and tabular victims now
measures a poisoning attack on a three-billion-parameter LLM
(`meta-llama/Llama-3.2-3B`), and neither the defense nor the poisoning metrics
were touched.
From `examples/attack_pipelines/run_text_backdoor.py`:

```python
from amulet.membership_inference.defenses import DPSGD

dp_training = DPSGD(
    model=dp_victim,               # a HFCausalLM, the same class the backdoor attacked
    criterion=criterion,
    optimizer=dp_optimizer,
    train_loader=train_loader,     # the poisoned loader
    device=device,
    delta=1e-5,
    max_per_sample_grad_norm=1.0,
    sigma=1.0,
    epochs=3,
)
dp_model = dp_training.train_private()
```

### The conformance test shaped ONION's interface

ONION purifies inputs, and the interface it would naturally expose (a
`purify` method) differs from the retraining interface the other poisoning
defenses share.
`tests/test_api_conformance.py` requires the shared entry point, so ONION
retrains the victim on purified data through `train_robust`, and a user
substitutes one poisoning defense for another without changing the surrounding
pipeline.
ONION keeps `purify` as an extra public helper for test-time cleaning; it
exposes `train_robust` **in addition**, never instead (`amulet/poisoning/defenses/onion.py`).
This is the check that turns the design rule into a build failure: a defense
shipping only a bespoke method fails CI.

### Adding a risk, metric, or architecture

The steps for extending the library along each axis are worked out in the
contribution examples:

- `examples/extending_amulet/custom_risk.md` adds a new risk and its attack.
- `examples/extending_amulet/custom_metric.md` adds a metric.
- `examples/extending_amulet/custom_architecture.md` adds a model architecture
  that subclasses `AmuletModel` and implements `get_hidden`.
