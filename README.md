# FlagOmni

English | [中文](README_cn.md)

## Introduction

FlagOmni is part of [FlagOS](https://flagos.io/), an open-source system software stack for diverse AI chips. FlagOS connects models, systems, and chips through software, reducing the cost of porting and maintaining models across hardware platforms.

FlagOmni is a library of operators for omni-modality generation models (video, image, audio) implemented in [Triton](https://github.com/triton-lang/triton). It targets mainstream generation and inference frameworks, with reusable computation implementations that can be optimized to accelerate generation workloads across multiple hardware backends.

For generation framework developers, FlagOmni aims to provide Triton implementations aligned with the relevant interfaces and computational semantics, reducing repeated development and hardware adaptation work. For operator developers, FlagOmni provides a collaborative platform for organizing implementations, tests, and performance optimization around real generation workloads.

The project is under active development: documentation and component implementations are added incrementally, and component and hardware compatibility have not been fully validated.

## Available LightX2V operators

`flag_omni.lightx2v` exposes normalization, packed SwiGLU, Q/K norm+RoPE,
scaled residual, reference blending, rotary tables and `sage_attention`.
Model policy and framework hooks belong in `lightx2v-plugin-FL`, not this library.

SageAttention has one public entry, selected internally by device and numerical
contract. `sage_attention(q, k, v, pv_dtype="bf16")` supports raw BF16 QKV on
CUDA/MUSA through the portable implementation. Raw `pv_dtype="fp8"` currently
requires CUDA SM90 and a compatible FlagTree TLE build; unsupported combinations
fail before compute, never silently change precision or retry after execution.

The same entry accepts prequantized inputs with `pv_dtype="fp8"` and
`quantization={"format": "per_thread_int8_fp8", "query_scale": qs,
"key_scale": ks, "value_scale": vs}`. The current MP31 backend requires contiguous
NHD Q/K INT8 and V E4M3 `[1,34498,7,128]`, with FP32 scale shapes
`[1,7,17280]`, `[1,7,1080]`, `[1,7,128]`. Q scales index `(m//16)*8+m%8`, K scales
index `(n//128)*4+(n%8)//2`; V scales are per head/channel. This is a distinct
prequantized contract, not a raw-QKV-to-MP31 quantization implementation.
A compatible MP31 FlagTree TLE toolchain and MUSA SDK (`FLAGOMNI_MUSA_SDK`, default
`/usr/local/musa`) are required. Finite quantized values and finite positive scales
are caller preconditions, not scanned on the hot path.

For repeated prequantized calls, pass a caller-owned `workspace={}` and warm up
before graph capture. Keep the workspace and input storage alive, update input
values in place, and do not share it concurrently between streams. Output storage
is reused; clone retained results. `return_fp32=True` also returns the same-call
FP32 output for diagnostic checks. No global tensor cache is used.

## Installation

```bash
uv pip install -e .                  # runtime
uv pip install -e ".[test]"          # development and tests
```

## Design Goals

- **Generation computation in Triton**: Develop reusable operators for forward computation in generation model inference, including normalization, activation, embedding, and fusion kernels.
- **Fusion and performance optimization**: Optimize operator fusion, memory access, and parallel computation for generation workloads to reduce intermediate data transfers and execution overhead.
- **Multiple hardware backends**: Build on compatible Triton compilation and runtime environments to adapt functionality and tune performance across hardware backends.
- **Upstream collaboration**: Align with the interfaces, computational semantics, and usage requirements of target generation frameworks, and keep pace with upstream changes.
- **Verifiable implementations**: Provide correctness tests and performance benchmarks for operators, with clear applicability and validation scope.

## Framework Scope

| Framework | Areas of Focus | Status |
|-----------|---------------|--------|
| LightX2V | Operators for video/image generation inference (RMS normalization, SwiGLU activation, VAE fusions, rotary embeddings) | 🚧 In Progress |
| FastVideo | Operators for unified video generation inference and post-training | 🔮 Planned |
| xDiT | Operators for scalable Diffusion Transformer inference | 🔮 Planned |

These frameworks form the initial scope. Support will be documented by operator, data type, and hardware backend in the compatibility notes.

## Relationship with Related Projects

FlagOmni is developed in collaboration with operator libraries, compilers, and inference frameworks in the FlagOS ecosystem, with distinct responsibilities:

- **[FlagGems](https://github.com/flagos-ai/FlagGems)**: Provides general-purpose operator implementations for LLM inference. FlagOmni prioritizes collaboration with FlagGems on reusable foundational operators to reduce duplicate implementations.
- **FlagOmni**: Focuses on operators and fused computation for omni-modality generation models, organizing their implementations, tests, and performance optimization.
- **Generation frameworks and upstream components**: Provide use cases and references for interfaces and computational semantics. FlagOmni develops and validates Triton implementations for their key computation paths.

## Directory Structure

```text
FlagOmni/
├── src/flag_omni/
│   ├── lightx2v/                   # LightX2V generic Triton operators (norm, activation, fusion …)
│   │   └── ops/                    # Portable operator implementations
│   ├── runtime/
│   │   └── backend/                # Multi-hardware backends: vendor → framework
│   │       ├── _nvidia/
│   │       │   └── lightx2v/       # NVIDIA specializations
│   │       └── _mthreads/
│   │           └── lightx2v/       # MThreads specializations
│   ├── testing/                    # Shared test utilities and fixtures
│   └── utils/                      # Device detection, config loading, type helpers
├── tests/
│   └── lightx2v/                   # Contract and numerical tests
├── benchmark/
│   └── lightx2v/                   # Latency and throughput benchmarks
├── conf/
│   └── operators.yaml              # Operator registry metadata
├── docs/                           # Documentation
├── tools/                          # CI scripts, GPU detection, code generation
├── CONTRIBUTING.md
├── SECURITY.md
├── LICENSE
└── pyproject.toml
```

Generic operators live in `src/flag_omni/<framework>/ops/`, hardware specializations in `runtime/backend/_<vendor>/<framework>/`. Each framework and hardware backend stays fully isolated.

### Adding a New Framework

Using FastVideo as an example, adding a new generation framework requires the following steps:

1. **Generic operators**: Create `src/flag_omni/fastvideo/` to hold the framework's Triton operator implementations
2. **Hardware specializations**: Create sub-directories under each hardware backend that needs specialization
   - `src/flag_omni/runtime/backend/_nvidia/fastvideo/`
   - `src/flag_omni/runtime/backend/_mthreads/fastvideo/`
3. **Tests and benchmarks**: Create the corresponding test and benchmark directories
   - `tests/fastvideo/`
   - `benchmark/fastvideo/`
4. **Operator registration**: Add operator entries in `conf/operators.yaml` with `from: [fastvideo]` to indicate the source framework

Adding a new framework does not require modifying any existing framework's code — each framework's operator implementations are fully isolated.

## Development and Validation

Each operator implementation should include a corresponding reference implementation, test cases, and a method for evaluating performance.

### Correctness Testing

- Cover representative input shapes, data types, and edge cases.
- Validate results against the reference implementation and specify numerical error tolerances.
- Record the software environment and hardware backend used in testing.

### Performance Testing

- Compare implementations under the same hardware, input, and precision conditions.
- Describe the reference implementation, warm-up procedure, and timing method.
- Report execution time, throughput, or device memory usage as appropriate for the computation.

## Contributing

Contributions to generation operators, framework adaptation, performance optimization, tests, and documentation are welcome.

- See the [contribution guide](CONTRIBUTING.md) for the contribution process.
- Report issues and suggest features through [GitHub Issues](https://github.com/flagos-ai/FlagOmni/issues).
- Report security issues privately by following the [security reporting instructions](SECURITY.md).

## License

This project is licensed under the [Apache License 2.0](LICENSE). Any included third-party code must retain its original license and copyright notices.
