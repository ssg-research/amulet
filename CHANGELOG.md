# Changelog

## [Unreleased]


### Features

* **poisoning:** textual backdoor attack with `TextBadNets`, the LoRA `HFCausalLM` victim, an ONION text purifier, and Hugging Face text loaders ([#105](https://github.com/ssg-research/amulet/issues/105)) ([eab3fa6](https://github.com/ssg-research/amulet/commit/eab3fa6c5447a9c936c2b3ecfe5f519b2d53dc88))
* **dp:** opt-in `BatchMemoryManager` for DP-SGD ([#108](https://github.com/ssg-research/amulet/issues/108)) ([ca1f65a](https://github.com/ssg-research/amulet/commit/ca1f65a1e930b6de60f9296ee4b11bc4dc8c477c))


### Bug Fixes

* **poisoning:** unify the poisoning defense ABC, give ONION a real LLM, and batch its scoring ([#107](https://github.com/ssg-research/amulet/issues/107)) ([3b356b8](https://github.com/ssg-research/amulet/commit/3b356b814ddef555e7df08b09abe1a51ea323073))

## [0.5.3](https://github.com/ssg-research/amulet/compare/v0.5.2...v0.5.3) (2026-07-06)


### Bug Fixes

* correct subgroup metrics crash and overhaul test suite ([#99](https://github.com/ssg-research/amulet/issues/99)) ([a6f0305](https://github.com/ssg-research/amulet/commit/a6f0305f7d28b07e640b3622c568c71182ceefd7))

## [0.5.2](https://github.com/ssg-research/amulet/compare/v0.5.1...v0.5.2) (2026-06-24)


### Bug Fixes

* **ci:** resolve dependabot alerts and pre-commit failure on release ([#94](https://github.com/ssg-research/amulet/issues/94)) ([dfaccf2](https://github.com/ssg-research/amulet/commit/dfaccf221b8d52e14e2e5483e8b3b2a3970cb15e))

## [0.5.1](https://github.com/ssg-research/amulet/compare/v0.5.0...v0.5.1) (2026-05-26)

### Bug Fixes

- **deps:** bump vulnerable dependencies (security) ([#89](https://github.com/ssg-research/amulet/issues/89)) ([34825e6](https://github.com/ssg-research/amulet/commit/34825e6fdaa8cfd27cc04864e55769106402b15a))
