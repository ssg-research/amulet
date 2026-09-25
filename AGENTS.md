# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

Scarab is a PyTorch-based research library for evaluating **unintended interactions** among ML defenses and risks across security, privacy, and fairness.
The central use case is composing an attack from one risk with a defense designed for another risk and measuring how they interfere.

Requires Python ~=3.11.0. Torch is selected via a hardware-specific extra (`cpu`, `cu128`, or `cu130`); see [Optional extras](#optional-extras).

## Where to find things

- **Dev setup, deps, lint/typecheck config:** [`pyproject.toml`](pyproject.toml) and [`.pre-commit-config.yaml`](.pre-commit-config.yaml)
- **Risk modules:** `scarab/<risk>/`, each with `attacks/`, `defenses/`, and optionally `metrics/` subpackages
  - Security: [`evasion/`](scarab/evasion/), [`poisoning/`](scarab/poisoning/), [`unauth_model_ownership/`](scarab/unauth_model_ownership/)
  - Privacy: [`membership_inference/`](scarab/membership_inference/), [`attribute_inference/`](scarab/attribute_inference/), [`distribution_inference/`](scarab/distribution_inference/), [`data_reconstruction/`](scarab/data_reconstruction/)
  - Fairness: [`discriminatory_behavior/`](scarab/discriminatory_behavior/)
- **Shared training/eval utilities:** [`scarab/utils/`](scarab/utils/). Check here before implementing your own helpers. If a needed utility is missing, add it to `scarab/utils/` and submit a PR. Functionality useful in one risk module is likely useful elsewhere.
- **Dataset loaders:** [`scarab/datasets/`](scarab/datasets/)
- **Model base class and architectures:** [`scarab/models/`](scarab/models/)
- **Runnable pipelines:** [`examples/attack_pipelines/`](examples/attack_pipelines/) and [`examples/defense_pipelines/`](examples/defense_pipelines/)
- **Extending Scarab (custom modules, metrics, risks):** [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) and [`examples/extending_scarab/`](examples/extending_scarab/)

## Commands

Dependency management is via `uv` (never `pip`/`conda`):

```bash
# Pick ONE torch build extra matching your hardware (check with `nvidia-smi`).
# Do NOT use `uv sync --all-extras`: the cpu/cu128/cu130 extras are declared
# conflicting, so requesting all of them errors.
uv sync --extra cu128 --extra dev   # CUDA 12.x box + dev tools
uv sync --extra cu130 --extra dev   # CUDA 13 box + dev tools
uv sync --extra cpu --extra dev     # GPU-less / CI: skip the CUDA runtime download
uv sync --extra cu128 --extra llm   # add the LLM stack (transformers/peft/datasets)
uv add <pkg>                        # add runtime dep
uv add --dev <pkg>                  # add dev dep
uv lock                             # regenerate uv.lock after editing pyproject.toml
```

Lint / typecheck / format (run via pre-commit so configuration stays in sync with CI):

```bash
uv run pre-commit install          # one-time
uv run pre-commit run --all-files  # ALWAYS use --all-files; omitting it only checks staged files
uv run ruff check --fix .
uv run ruff format .
uv run basedpyright                # standard mode; venv is .venv (configured in pyproject.toml)
```

Tests live in `tests/`, organized by module (`tests/<risk>/` per risk, plus `models/`, `datasets/`, `utils/`, and the top-level `tests/test_api_conformance.py`).
They are tiered by pytest markers (`integration`, `gpu`, `slow`) declared in `[tool.pytest.ini_options]`.
The default `addopts` deselects all three (`-m 'not integration and not gpu and not slow'`), so a bare `uv run pytest` runs only the fast, unmarked tests.
Run examples as end-to-end smoke tests:

```bash
uv run python examples/get_started.py
uv run python examples/attack_pipelines/run_evasion.py
```

## Non-obvious rules

### API contract

Attacks and defenses must not emit metrics.
They return outputs (e.g. adversarial `DataLoader`, defended `nn.Module`) consumed by metrics in `scarab/utils/__metrics.py` or the risk's own `metrics/`.

Each risk has an ABC base class in `scarab/<risk>/attacks/` and `scarab/<risk>/defenses/`.
**Every defense must implement its risk's training-shaped entry-point method.** This is a
hard convention, not a suggestion, and it is enforced by `tests/test_api_conformance.py`
(a defense exposing only a bespoke method fails CI). The standard entry-point methods are:

| Role                           | Method                                             |
| ------------------------------ | -------------------------------------------------- |
| All attacks (except poisoning) | `attack()`                                         |
| Poisoning attacks              | `poison_train(dataset)` and `poison_test(dataset)` |
| Evasion + poisoning defenses   | `train_robust()`                                   |
| Membership inference defense   | `train_private()`                                  |
| Fairness defense               | `train_fair()`                                     |
| Watermarking defense           | `watermark()`                                      |
| Fingerprinting defense         | `fingerprint()`                                    |

A defense may expose extra public helpers in addition to its entry point, never instead
of it. `ONION` is the reference case: it is a poisoning defense subclassing
`PoisoningDefense` alongside `OutlierRemoval`, so it implements `train_robust()` (purify the
poisoned training data, then retrain the target), and it also exposes
`purify(dataset)` for cleaning inputs at test time. Do not add a defense on a bespoke base
class with a non-standard entry point; reshape the shared base instead. The textual backdoor
attack `TextBadNets` and the LoRA-LLM target `HFCausalLM` need the optional `llm` extra; see
[Optional extras](#optional-extras).

Some classes expose additional public helpers for experimentation. For example, `MembershipInferenceAttack` has `train_shadow_model()` / `prepare_shadow_models()` and `DistributionInferenceAttack` has `train_model_population()` / `prepare_model_populations()`.
Check the base class before assuming `attack()` is the only callable.

### Models

Any model under `scarab/models/` must subclass `ScarabModel` ([`scarab/models/base.py`](scarab/models/base.py)) and implement `get_hidden(self, x) -> Tensor`.
Several modules depend on intermediate activations; omitting `get_hidden` breaks them silently.
Match the base signature's parameter name `x` on both `forward` and `get_hidden` (basedpyright enforces override compatibility), even when the input is token ids rather than pixels.

`HFCausalLM` ([`scarab/models/hf_causal_lm.py`](scarab/models/hf_causal_lm.py)) is the reference example of subclassing `ScarabModel` around a real pretrained backbone: a LoRA-adapted HuggingFace **causal (decoder-only) LM** (Llama, GPT-2, Mistral, …) that keeps its generative base. One shared adapted decoder backs three roles: classification (`forward` returns the bare logits tensor, not the `SequenceClassifierOutput`, so `train_classifier`, `DPSGD.train_private`, and `get_accuracy` drive it unchanged), perplexity scoring (`perplexity`, what `ONION` consumes, since the target itself is the reference LM), and generation (`generate`). Encoder-only (BERT) and seq2seq (T5) models do not fit and are out of scope. It needs the `llm` extra.

`initialize_model` uses a central capacity map and only covers the built-in CNNs; models whose constructors do not fit its `(arch, capacity, num_features, num_classes)` signature (e.g. `HFCausalLM`) are constructed directly. See #104.

`WatermarkNN` and `DatasetInference` have separate ABC base classes (`WatermarkDefense`, `FingerprintDefense`).
There is no shared parent. This is intentional.

### Datasets

Image loaders follow a 3-step fallback to ensure availability:

1. **Processed local** (e.g. `celeba.npz`, `lfw_images.npz`)
2. **Raw local** (e.g. `img_align_celeba/`, `lfw_home/`)
3. **GDrive download**: IDs are hard-coded in [`scarab/datasets/__image_datasets.py`](scarab/datasets/__image_datasets.py) (similar to how PyTorch ships dataset URLs), so no configuration is needed.

Text loaders ([`scarab/datasets/__text_datasets.py`](scarab/datasets/__text_datasets.py): `load_sst2`, `load_agnews`, `load_imdb`) instead pull from the Hugging Face hub via `datasets` into a project-local `./data/<name>` cache. That divergence is intentional: HF manages text corpora and their splits. They return a `ScarabDataset` with `modality="text"` whose `train_set`/`test_set` are `TextTensorDataset` instances, a `TensorDataset` of padded `input_ids` that also carries the raw `.texts` (so ONION can re-score perplexity before the target tokenizer runs) and the `tokenizer_name`. `ScarabDataset.modality` is `Literal["image", "tabular", "text"]`. Text loaders need the `llm` extra.

### Optional extras

- **Torch build (`cpu` / `cu128` / `cu130`):** mutually exclusive (declared in `[tool.uv] conflicts`), each pinning the same `torch`/`torchvision` but routed to the matching PyTorch index via `[tool.uv.sources]`. Always sync with exactly one. The base `torch`/`torchvision` floor stays loose; the extras exist so a `uv sync` produces a driver-correct GPU build instead of a cu13 wheel that silently runs on CPU.
- **`llm`:** the Hugging Face stack (`transformers`, `peft`, `accelerate`, `datasets`) for the textual backdoor pipeline (`TextBadNets`, `HFCausalLM`, `ONION`, the text loaders). Kept optional so the base install stays lean and the macOS dev machine / fast CI tier never pull it. Every HF import is lazy and guarded, so `import scarab` works without the extra and constructing an LLM component without it raises a clear "sync with the llm extra" error.
- **`bitsandbytes`** (4-bit load path in `HFCausalLM`) is GPU/Linux-only and deliberately **not** in the `llm` extra. Its import is guarded, off by default, and never used under DP (Opacus per-sample hooks do not compose with 4-bit layers).

### Tooling

- Keep the `ruff-pre-commit` hook rev in sync with `ruff==` in `pyproject.toml`. A mismatch silently skips rules.
- `B903` (class-could-be-dataclass) is globally ignored. Base classes that provide shared state for subclasses are a valid pattern here.
- Pandas stubs: use `# type: ignore[reportArgumentType]` for `columns=list[str]` and `# type: ignore[reportAttributeAccessIssue]` for `.isin()`. Do not use `cast()`, an established repo convention.
- Dependency versions are pinned exactly. `cleverhans`, `opacus`, and `captum` are sensitive to version drift. Do not loosen pins without a reason.
- The distribution name is `scarabml`; the import name is `scarab`.
- Docstrings use Google style: imperative summary line, `Args:` / `Returns:` / `Raises:` sections, no type repetition from the signature, no RST markup.
