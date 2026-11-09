# Contributing to FlagOmni

FlagOmni follows the
[FlagOS contribution guidelines](https://github.com/flagos-ai/community/blob/main/contributors/CONTRIBUTING.md)
and [Code of Conduct](https://github.com/flagos-ai/community/blob/main/CODE_OF_CONDUCT.md).
This repository is in its initial setup stage. Contributions to Triton generation
operators, framework support, tests, benchmarks, and documentation are welcome.

## Propose a change

Search existing issues and pull requests before opening a new one. Describe the
environment, expected behavior, and reproduction steps for a bug. Discuss substantial changes before
implementation; cross-project proposals should follow the
[FlagOS FEP process](https://github.com/flagos-ai/community/tree/main/fep).

Report security vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

## Work on a branch

Fork and clone the repository, or use your existing checkout if you have write
access. Create a separate branch for each change. Keep changes focused and update
relevant documentation alongside user-facing behavior.

## Implement and validate

Place operator implementations under `src/flag_omni/ops/` for generic Triton kernels,
and `src/flag_omni/runtime/backend/_<vendor>/` for hardware-specific implementations.
Keep corresponding tests and benchmarks in `tests/` and `benchmark/`.

Run the full test suite before opening a pull request:

```bash
uv pip install -e ".[test]"
pytest
```

Pre-commit hooks enforce formatting and import order:

```bash
pre-commit install
pre-commit run --all-files
```
