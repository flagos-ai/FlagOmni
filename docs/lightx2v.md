# LightX2V operators

## Introduction

FlagOmni provides LightX2V-specific operators implemented with Triton
and FlagTree, with generic implementations and vendor/architecture backends.

## Relationship with FlagGems and lightx2v-plugin-FL

- [FlagGems](https://github.com/flagos-ai/FlagGems) maintains general-purpose
  operators and shared backend utilities.
- **FlagOmni** exposes LightX2V-specific numerical operators through the
  `flag_omni.lightx2v` Python namespace.
- [lightx2v-plugin-FL](https://github.com/ModelTC/lightx2v-plugin-FL) owns model
  configuration, loading, scheduling, communication and operator integration.

## Operators

| API | Contract |
|---|---|
| `rms_norm` | BF16 D128/D5376; FP32 weight multiplication before the final cast |
| `packed_swiglu` | Packed `[value, gate]`; input-precision SiLU before multiplication |
| `qk_norm_rope` | D64 Q/K normalization and 48-channel split rotary embedding |
| `scaled_residual` | FP32 state/scale with FP16 update |
| `residual_rms_norm` | D2048 FP32 residual/RMS with FP16 normalized output |
| `reference_blend` | FP32 reference-blend update |
| `rotary_tables` | SM90 rotary position tables |

The RMS `tier` selects ordinary Triton (`baseline`), MTT S5000 TLE (`tle`),
or NVIDIA SM90 Gluon/PTX (`sm90`). Advanced tiers require a compatible compiler.
Other operators use their ordinary Triton implementations; rotary tables require
SM90. See the public API docstrings for supported dtypes, layouts and arguments.

## Quick Installation

Install the device-matched PyTorch, Triton/FlagTree builds first. FlagGems is not a runtime dependency.

```bash
python -m pip install --no-deps -e .
python -m pip install pytest numpy
```

## Usage Example

```python
import torch
import flag_omni.lightx2v

x = torch.randn((44, 5376), device="cuda", dtype=torch.bfloat16)
weight = torch.ones(5376, device=x.device, dtype=x.dtype)
y = flag_omni.lightx2v.rms_norm(x, weight, eps=1e-5, tier="baseline")
```

SageAttention adds an ordinary Triton BF16-PV baseline, the final NVIDIA SM90
TLE FP8-PV full-call implementation, and an explicit final S5000 prequantized
TLE operator. Their different numerical contracts and compiler requirements
are documented in [SageAttention](sage_attention.md).

## Tests and Benchmarks

Tests and benchmarks use per-operator pytest files and markers.

```bash
# Check test discovery without executing device workloads.
python -m pytest tests benchmark --collect-only -q

# CPU API contracts (accelerator tests are skipped without a selected device).
python -m pytest tests/lightx2v/test_contracts.py -q

# Device numerical tests; fail if the requested accelerator is unavailable.
python -m pytest tests --accelerator-device cuda:0 --require-accelerator -q

# Operator benchmarks, including allocation and five explicit warmup calls.
python -m pytest benchmark --accelerator-device cuda:0 --require-accelerator \
  --tier sm90 --warmup 25 --iter 100 -s
```

Use `--accelerator-device musa:0 --tier tle` for the S5000 RMS benchmark.
Timing warmup and sample durations are in milliseconds. The RMS benchmark
reference is an unfused Torch expression with weight multiplication before the
final cast. The large-address SwiGLU test allocates the 1080p packed shape.

## License

This project is licensed under the [Apache 2.0 License](../LICENSE).

## Code ownership and layout

`lightx2v` is a framework namespace, not a model or hardware namespace:

```text
src/flag_omni/
  lightx2v/
    __init__.py                 # stable public exports
    ops/
      api.py                    # public numerical contracts
      reference_blend.py
      rms_norm/                 # portable baseline and helpers
      fusions/                  # swiglu, qk_norm_rope, residual, residual_norm
      sage_attention/           # API, baseline, quantization, math
  runtime/
    op_registrar.py             # lazy public lookup and kernel dispatch
    sage_attention_dispatch.py # Sage platform selection
    backend/
      _nvidia/lightx2v/hopper/  # NVIDIA SM90 implementation
      _mthreads/lightx2v/       # vendor tuning and implementations
        mp31/                   # S5000/MP31 prequantized wrapper and core
```

Model integration belongs in [lightx2v-plugin-FL](https://github.com/flagos-ai/lightx2v-plugin-FL),
not this operator repository. FlagOmni has no model adapter directory. H3 policy
names, registry wiring, weight staging and lifecycle hooks stay in plugin PR #8;
that plugin maps its `h3_*` policies to the operation-named public APIs above.
Vendor checks, launch tuning, SDK handling and architecture kernels belong in the
runtime/backend hierarchy. Portable APIs request runtime dispatch rather than
importing a specific vendor implementation.

This initial, unmerged API replaces the provisional `h3_*` exports with the names
in the API table. The plugin pins the matching full FlagOmni commit, so neither
repository relies on a model-specific operator API. The explicit
`sage_attention_s5000_quantized` contract remains available; its implementation
and preparation API live in the MP31 backend. Private module paths changed;
consumers should use the top-level public APIs.
