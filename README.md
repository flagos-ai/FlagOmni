# FlagOmni

English | [中文](README_cn.md)

## Introduction

FlagOmni is part of [FlagOS](https://flagos.io/), an open-source system software stack for diverse AI chips. FlagOS connects models, systems, and chips through software, reducing the cost of porting and maintaining models across hardware platforms.

FlagOmni is a library of operators for omni-modality generation models (video, image, audio) implemented in [Triton](https://github.com/triton-lang/triton). It targets mainstream generation and inference frameworks, with reusable computation implementations that can be optimized to accelerate generation workloads across multiple hardware backends.

For generation framework developers, FlagOmni aims to provide Triton implementations aligned with the relevant interfaces and computational semantics, reducing repeated development and hardware adaptation work. For operator developers, FlagOmni provides a collaborative platform for organizing implementations, tests, and performance optimization around real generation workloads.

The project is under active development: documentation and component implementations are added incrementally, and component and hardware compatibility have not been fully validated.

## Supported Frameworks

| Framework | Status | Description |
|-----------|--------|-------------|
| LightX2V | 🚧 In Progress | Lightweight video/image generation inference framework |
| FastVideo | 🔮 Planned | Unified video generation inference and post-training framework |
| xDiT | 🔮 Planned | Scalable inference engine for Diffusion Transformers |

## Installation

```bash
uv pip install -e .                  # runtime
uv pip install -e ".[test]"          # development and tests
```

## Relationship with Other FlagOS Projects

- **FlagGems**: General-purpose operator library for LLM inference
- **FlagGems-vllm / FlagGems-sglang**: Operator libraries for specific LLM inference frameworks
- **FlagOmni**: Operator library for omni-modality generation models — complementary to FlagGems

## License

Apache License 2.0
