# Contributing to Scarab

Thank you for your interest in contributing to Scarab! This guide will help you get started with the contribution process.

## Reporting Issues

We use GitHub Issues to track bugs and feature requests.

- **Bug Reports**: Please include a clear title, a description of the issue, and steps to reproduce the bug. Providing a minimal code snippet or a failing test case is highly recommended.
- **Feature Requests**: We welcome new ideas! For research-related features (new risks or defenses), please include a reference to the relevant peer-reviewed paper.

## Contributing Code

### Environment Setup

Scarab uses `uv` for dependency management.

1. **Install uv** by following the installation guide in the uv documentation.

2. **Sync** from the repository root:

   Pick one torch build extra matching your hardware (`cpu`, `cu128`, or `cu130`; check with `nvidia-smi`). They are mutually exclusive, so `uv sync --all-extras` is not valid.

   ```bash
   uv sync --extra cu128 --extra dev   # or --extra cpu / --extra cu130
   # add --extra llm for the text/LLM stack (transformers, peft, datasets)
   ```

3. **Install Pre-commit**:

   ```bash
   uv run pre-commit install
   ```

### Coding Standards

- **Formatting**: We use `ruff` for formatting and linting. Run `uv run ruff format .` and `uv run ruff check --fix .`.
- **Type Checking**: We use `basedpyright` for static type analysis. Run `uv run basedpyright`.
- **Docstrings**: Use Google-style docstrings.
- **Naming**: Internal utility modules are prefixed with underscores (e.g., `__base.py`). Private class members should follow the `_name` convention.

### Adding a New Risk Module

1. **Subclass the Base Class**: Each risk has a base class (e.g., `EvasionAttack`). Your new module must subclass this.
2. **Implementation**: Place your file in the appropriate directory: `scarab/<risk>/<attacks|defenses|metrics>/new_module.py`.
3. **API Contract**:
   - **Attacks**: Implement an `attack()` method (except for poisoning, which uses `poison_train` and `poison_test`).
   - **Defenses**: Every defense **must** implement the training-shaped entry point for its risk: poisoning/evasion → `train_robust()`, membership inference → `train_private()`, fairness → `train_fair()`, ownership → `watermark()` / `fingerprint()`. This is enforced by `tests/test_api_conformance.py`: a defense that ships only a bespoke method (e.g. a `purify`-only input cleaner) fails the suite. A defense may expose extra public helpers **in addition to** its entry point, never instead of it, and should subclass its risk's existing defense base rather than introducing a new one.
   - **Return Values**: Attacks/defenses should return artifacts (e.g., a `DataLoader` or a `nn.Module`), not metrics directly.
4. **Export**: Import your class in the risk's `__init__.py`.
5. **Example**: Add a runnable script in `examples/attack_pipelines/` or `examples/defense_pipelines/`.

### Adding a Dataset

Datasets should return a `ScarabDataset` dataclass.

```python
@dataclass
class ScarabDataset:
    train_set: Dataset
    test_set: Dataset
    num_features: int
    num_classes: int
    modality: Literal["image", "tabular", "text"]
    sensitive_columns: list[str] | None = None
    x_train: np.ndarray | None = None
    ...
```

1. Implement the loading logic in `scarab/datasets/__image_datasets.py`, `scarab/datasets/__tabular_datasets.py`, or `scarab/datasets/__text_datasets.py` (text corpora load from the Hugging Face hub and return `TextTensorDataset` instances with `modality="text"`).
2. Update `load_data` in `scarab/utils/__pipeline.py` to support the new dataset.

### Adding a Model Architecture

1. Define your architecture in a new file under `scarab/models/`.
2. Ensure it implements a `get_hidden(self, x)` method.
3. Update `initialize_model` in `scarab/utils/__pipeline.py` to include the new architecture.

## Pull Request Process

1. Create a new branch for your feature or bug fix.
2. Ensure all tests pass: `uv run pytest`.
3. Run linting and type checking.
4. Submit your PR with a descriptive title and summary of changes.
