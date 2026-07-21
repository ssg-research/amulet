# Amulet

[![CI Status](https://github.com/ssg-research/amulet/actions/workflows/precommit-main.yaml/badge.svg)](https://github.com/ssg-research/amulet/actions/workflows/precommit-main.yaml)
[![CodeQL](https://github.com/ssg-research/amulet/actions/workflows/codeql.yaml/badge.svg)](https://github.com/ssg-research/amulet/actions/workflows/codeql.yaml)
[![PyPI version](https://img.shields.io/pypi/v/amuletml.svg)](https://pypi.org/project/amuletml/)
[![Python Versions](https://img.shields.io/pypi/pyversions/amuletml.svg)](https://pypi.org/project/amuletml/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Amulet is a Python machine learning (ML) package to evaluate the susceptibility of different risks to security, privacy, and fairness. Amulet is applicable to evaluate how algorithms designed to reduce one risk may impact another unrelated risk and compare different attacks/defenses for a given risk.

Amulet builds upon prior work titled [“SoK: Unintended Interactions among Machine Learning Defenses and Risks”](https://arxiv.org/abs/2312.04542) which appears in IEEE Symposium on Security and Privacy 2024. The SoK covers only two interactions and identifies the design of a software library to evaluate unintended interactions as future work. Amulet addresses this gap by including eight different risks each covering their own attacks, defenses and metrics.

Amulet is:

- Comprehensive: Covers the most representative attacks/defenses/metrics for different risks.
- Extensible: Easy to include additional risks, attacks, defenses, or metrics.
- Consistent: Allows using different attacks/defenses/metrics with a consistent, easy-to-use API.
- Applicable: Allows evaluating unintended interactions among defenses and attacks.

Built to work with PyTorch, you can incorporate Amulet into your current ML pipeline to test how your model interacts with these state-of-the-art defenses and risks. Alternatively, you can use the example pipelines to bootstrap your pipeline.

## Getting Started

The easiest way to start is via `pip`:

`pip install amuletml`

For a reproducible GPU build (and for development), install from source with `uv` and select the torch build matching your driver: `uv sync --extra cu128` (CUDA 12.x), `uv sync --extra cu130` (CUDA 13), or `uv sync --extra cpu`. Add `--extra llm` for the optional text/LLM stack (used by the textual backdoor pipeline).

### Test installation

To test your installation, please run [examples/get_started.py](https://github.com/ssg-research/amulet/blob/main/examples/get_started.py). This script also serves as a starting point to learn how to use the library.

### Learn More

For more information on the basics about the library, please see the [Getting Started guide](https://github.com/ssg-research/amulet/blob/main/docs/GETTING_STARTED.md).

To see the attacks, defenses, and risks (modules) that Amulet implements, please refer to the [Module Hierarchy](https://github.com/ssg-research/amulet/blob/main/docs/module_guide/1_INTRO.md).

For each module, please see [amulet/examples](https://github.com/ssg-research/amulet/tree/main/examples) for implementations of pipelines that include recommendations on how to run each module.

### Contributing

See [CONTRIBUTING](https://github.com/ssg-research/amulet/blob/main/docs/CONTRIBUTING.md) for guidance.
