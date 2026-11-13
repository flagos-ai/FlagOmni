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
| `h3_rms_norm` | BF16 D128/D5376; FP32 weight multiplication before the final cast |
| `h3_swiglu` | Packed `[value, gate]`; input-precision SiLU before multiplication |
| `h3_vae_qk_norm_rope` | D64 Q/K normalization and 48-channel split rotary embedding |
| `h3_vae_residual` | FP32 state/scale with FP16 update |
| `h3_vae_residual_norm` | D2048 FP32 residual/RMS with FP16 normalized output |
| `h3_reference_blend` | FP32 reference-blend update |
| `h3_rotary_tables` | SM90 rotary position tables |

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
y = flag_omni.lightx2v.h3_rms_norm(x, weight, eps=1e-5, tier="baseline")
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
    models/h3.py                # H3-specific numerical contracts/adapters
    ops/
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

Model names belong in adapters/contracts, not in the portable operator directory.
Vendor checks, launch tuning, SDK handling and architecture kernels belong in the
runtime/backend hierarchy. Portable operator APIs request runtime dispatch rather
than importing a specific vendor implementation. Tests may name a hardware target
when they verify that backend's explicit contract.

Existing top-level `flag_omni.lightx2v.h3_*` and
`sage_attention_s5000_quantized` exports remain compatibility APIs. The latter is
an explicit prequantized contract, not an automatic replacement for raw-QKV
SageAttention. Its implementation and preparation API now live at
`flag_omni.runtime.backend._mthreads.lightx2v.mp31.sage_attention`.
Private module paths changed; import operators through the top-level public API.
No model loading, patching or scheduling is introduced by `models/h3.py`.
