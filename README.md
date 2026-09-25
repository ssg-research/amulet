# Scarab

> **Artifact reviewers:** start at [`artifact/ARTIFACT.md`](artifact/ARTIFACT.md). It is the reproduction harness and reviewer entry point for the benchmark submission.

Scarab is a Python machine learning (ML) package to evaluate the susceptibility of different risks to security, privacy, and fairness. Scarab is applicable to evaluate how algorithms designed to reduce one risk may impact another unrelated risk and compare different attacks/defenses for a given risk.

Scarab includes eight different risks, each covering their own attacks, defenses and metrics.

Scarab is:

- Comprehensive: Covers the most representative attacks/defenses/metrics for different risks.
- Extensible: Easy to include additional risks, attacks, defenses, or metrics.
- Consistent: Allows using different attacks/defenses/metrics with a consistent, easy-to-use API.
- Applicable: Allows evaluating unintended interactions among defenses and attacks.

Built to work with PyTorch, you can incorporate Scarab into your current ML pipeline to test how your model interacts with these state-of-the-art defenses and risks. Alternatively, you can use the example pipelines to bootstrap your pipeline.

## Getting Started

Install from source with `uv` and select the torch build matching your driver: `uv sync --extra cu128` (CUDA 12.x), `uv sync --extra cu130` (CUDA 13), or `uv sync --extra cpu`. Add `--extra llm` for the optional text/LLM stack (used by the textual backdoor pipeline).

### Test installation

To test your installation, please run [examples/get_started.py](examples/get_started.py). This script also serves as a starting point to learn how to use the library.

### Learn More

For more information on the basics about the library, please see the [Getting Started guide](docs/GETTING_STARTED.md).

To see the attacks, defenses, and risks (modules) that Scarab implements, please refer to the [Module Hierarchy](docs/module_guide/1_INTRO.md).

For each module, please see [examples/](examples/) for implementations of pipelines that include recommendations on how to run each module.

### Contributing

See [CONTRIBUTING](docs/CONTRIBUTING.md) for guidance.
