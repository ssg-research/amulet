# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

Amulet (`amuletml` on PyPI) is a PyTorch-based research library for evaluating **unintended interactions** among ML defenses and risks across security, privacy, and fairness.
It builds on "SoK: Unintended Interactions among Machine Learning Defenses and Risks" (IEEE S&P 2024).
The central use case is composing an attack from one risk with a defense designed for another risk and measuring how they interfere.

Requires Python ~=3.11.0 and (for GPU) CUDA 11.8+ with PyTorch 2.2+.

## Where to find things

- **Dev setup, deps, lint/typecheck config:** [`pyproject.toml`](pyproject.toml) and [`.pre-commit-config.yaml`](.pre-commit-config.yaml)
- **Risk modules:** `amulet/<risk>/` — each has `attacks/`, `defenses/`, and optionally `metrics/` subpackages
  - Security: [`evasion/`](amulet/evasion/), [`poisoning/`](amulet/poisoning/), [`unauth_model_ownership/`](amulet/unauth_model_ownership/)
  - Privacy: [`membership_inference/`](amulet/membership_inference/), [`attribute_inference/`](amulet/attribute_inference/), [`distribution_inference/`](amulet/distribution_inference/), [`data_reconstruction/`](amulet/data_reconstruction/)
  - Fairness: [`discriminatory_behavior/`](amulet/discriminatory_behavior/)
- **Shared training/eval utilities:** [`amulet/utils/`](amulet/utils/) — check here before implementing your own helpers. If a needed utility is missing, add it to `amulet/utils/` and submit a PR — functionality useful in one risk module is likely useful elsewhere.
- **Dataset loaders:** [`amulet/datasets/`](amulet/datasets/)
- **Model base class and architectures:** [`amulet/models/`](amulet/models/)
- **Runnable pipelines:** [`examples/attack_pipelines/`](examples/attack_pipelines/) and [`examples/defense_pipelines/`](examples/defense_pipelines/)
- **Extending Amulet (custom modules, metrics, risks):** [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) and [`examples/extending_amulet/`](examples/extending_amulet/)

## Commands

Dependency management is via `uv` (never `pip`/`conda`):

```bash
uv sync                     # install runtime deps, create .venv
uv sync --all-extras        # include dev deps (ruff, basedpyright, pytest, pre-commit)
uv add <pkg>                # add runtime dep
uv add --dev <pkg>          # add dev dep
uv lock                     # regenerate uv.lock after editing pyproject.toml
```

Lint / typecheck / format (run via pre-commit so configuration stays in sync with CI):

```bash
uv run pre-commit install          # one-time
uv run pre-commit run --all-files  # ALWAYS use --all-files; omitting it only checks staged files
uv run ruff check --fix .
uv run ruff format .
uv run basedpyright                # standard mode; venv is .venv (configured in pyproject.toml)
```

Tests live in `tests/` (`unit/` and `integration/`).
Run examples as end-to-end smoke tests:

```bash
uv run python examples/get_started.py
uv run python examples/attack_pipelines/run_evasion.py
```

## Non-obvious rules

### API contract

Attacks and defenses must not emit metrics.
They return outputs (e.g. adversarial `DataLoader`, defended `nn.Module`) consumed by metrics in `amulet/utils/__metrics.py` or the risk's own `metrics/`.

Each risk has an ABC base class in `amulet/<risk>/attacks/` and `amulet/<risk>/defenses/`.
The standard entry-point methods are:

| Role                           | Method                                             |
| ------------------------------ | -------------------------------------------------- |
| All attacks (except poisoning) | `attack()`                                         |
| Poisoning attacks              | `poison_train(dataset)` and `poison_test(dataset)` |
| Evasion + poisoning defenses   | `train_robust()`                                   |
| Membership inference defense   | `train_private()`                                  |
| Fairness defense               | `train_fair()`                                     |
| Watermarking defense           | `watermark()`                                      |
| Fingerprinting defense         | `fingerprint()`                                    |

Some classes expose additional public helpers for experimentation — for example, `MembershipInferenceAttack` has `train_shadow_model()` / `prepare_shadow_models()` and `DistributionInferenceAttack` has `train_model_population()` / `prepare_model_populations()`.
Check the base class before assuming `attack()` is the only callable.

### Models

Any model under `amulet/models/` must subclass `AmuletModel` ([`amulet/models/base.py`](amulet/models/base.py)) and implement `get_hidden(self, x) -> Tensor`.
Several modules depend on intermediate activations; omitting `get_hidden` breaks them silently.

`WatermarkNN` and `DatasetInference` have separate ABC base classes (`WatermarkDefense`, `FingerprintDefense`).
There is no shared parent — this is intentional.

### Datasets

Loaders follow a 3-step fallback to ensure availability:

1. **Processed local** (e.g. `celeba.npz`, `lfw_images.npz`)
2. **Raw local** (e.g. `img_align_celeba/`, `lfw_home/`)
3. **GDrive download** — IDs are hard-coded in [`amulet/datasets/__image_datasets.py`](amulet/datasets/__image_datasets.py) (similar to how PyTorch ships dataset URLs), so no configuration is needed.

### Tooling

- Keep the `ruff-pre-commit` hook rev in sync with `ruff==` in `pyproject.toml`. A mismatch silently skips rules.
- `B903` (class-could-be-dataclass) is globally ignored. Base classes that provide shared state for subclasses are a valid pattern here.
- Pandas stubs: use `# type: ignore[reportArgumentType]` for `columns=list[str]` and `# type: ignore[reportAttributeAccessIssue]` for `.isin()`. Do not use `cast()` — established repo convention.
- Dependency versions are pinned exactly. `cleverhans`, `opacus`, and `captum` are sensitive to version drift. Do not loosen pins without a reason.
- The package is published to PyPI as `amuletml`; the import name is `amulet`.
- Docstrings use Google style: imperative summary line, `Args:` / `Returns:` / `Raises:` sections, no type repetition from the signature, no RST markup.
