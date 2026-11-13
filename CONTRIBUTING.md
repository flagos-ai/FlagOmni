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

Place generic Triton operators under `src/flag_omni/<framework>/ops/`,
model-specific numerical contracts/adapters under `<framework>/models/`, and
hardware-specific implementations under
`src/flag_omni/runtime/backend/_<vendor>/<framework>/`. Architecture-specific
implementations belong below the framework directory (for example,
`_nvidia/lightx2v/hopper/`). Keep corresponding tests and benchmarks in
`tests/<framework>/` and `benchmark/<framework>/`.

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
