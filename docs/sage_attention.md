# SageAttention

## Public interfaces and distinct numerical contracts

```python
from flag_omni import lightx2v

# Ordinary upstream Triton, no TLE/Gluon/TMA or vendor assembly.
y = lightx2v.sage_attention(q, k, v, tensor_layout="NHD",
                            pv_dtype="bf16", implementation="baseline")
# Final NVIDIA SM90 TLE pipeline, with full quantization and allocation.
y = lightx2v.sage_attention(q, k, v, tensor_layout="NHD",
                            pv_dtype="fp8", implementation="sm90")
```

The full-call API supports dense noncausal inference, NHD/HND tensors, equal
heads, D128, matching device/dtype and unit final stride. Empty dimensions,
causal attention, LSE, backward and unsupported precision choices fail before
computation. Masked and variable-length semantics are not implemented. Choosing
FP8 never silently changes to BF16, and execution errors never trigger fallback.

### Ordinary Triton baseline

BF16 inputs use per-thread INT8 Q/K and BF16 P/V with FP32 accumulation. K is
centered in the input dtype. Q has 32 scales per 64-token group; K has four
scales per 128-token group. Partial tiles are masked. The implementation is
independent of the optional TLE compiler and native SageAttention package.
Dense unquantized SDPA is not a same-arithmetic oracle.

### Final NVIDIA SM90 TLE

This is the frozen `sage_tle_pipeline_v4` implementation, not an earlier
experimental pipeline. The final wrapper uses the control path for NK <= 128
and KV single-buffer / register-cap 168 for longer keys. It retains native
K-mean reduction order, per-thread INT8 Q/K, per-channel E4M3 FP8 V and the
SM90 `fp32+fp32` accumulation contract. It supports FP16/BF16 input. The
noncontiguous V-stride quantization fix and zero-V handling are retained.

The independent native oracle is
`sageattn_qk_int8_pv_fp8_cuda_sm90(qk_quant_gran="per_thread",
pv_accum_dtype="fp32+fp32")`. The similarly named SM89 entry has a different
quantization/block geometry and is not a bitwise oracle. The native package is
needed only for reference tests and the paired benchmark, not implementation.
FlagTree TLE with SM90 TMA/WGMMA support is required for execution.

## Final MThreads S5000 / MP31 TLE

The final Triton core is provided separately as
`lightx2v.sage_attention_s5000_quantized`. It is explicitly a **prequantized,
fixed-shape** kernel, not a replacement for the full-call generic API. It does
not call MATE, and the older MATE quantization recipe is not interchangeable.
The R3473 and R3704/R3791 frozen Triton and layout source files are byte-identical.
This delivery uses only the qualified R3473 compiler options; it does not load,
bundle or require the later SDK14 custom native-ADD pass.

Input contract, all contiguous on the same MP31 MUSA device:

| Input | Shape | Type / meaning |
|---|---|---|
| Q | `[1,34498,7,128]` | NHD INT8 |
| K | `[1,34498,7,128]` | NHD INT8 |
| V | `[1,34498,7,128]` | NHD E4M3 FP8 (`torch.float8_e4m3fn`) |
| Q scale | `[1,7,17280]` | FP32; query `m` indexes `(m // 16) * 8 + m % 8` |
| K scale | `[1,7,1080]` | FP32; key `n` indexes `(n // 128) * 4 + (n % 8) // 2` |
| V scale | `[1,7,128]` | FP32 per head/channel |

Scales must contain finite positive quantization scales and Q/K/V must contain
finite quantized values. This data-value precondition is not scanned in the
hot path. No user-supplied softmax scale, causal mask or output dtype is accepted.
The frozen D128 scale is `0.127517431974411` (log2 convention), with FP8
probability offset `-8.807000160217285`. Output is NHD BF16; `diagnostic=True`
also returns the FP32 values from the same invocation before the RNE BF16 cast.

```python
# qi/ki/vf and qs/ks/vs are already quantized with the above contract.
y = lightx2v.sage_attention_s5000_quantized(qi, ki, vf, qs, ks, vs)
y, fp32 = lightx2v.sage_attention_s5000_quantized(
    qi, ki, vf, qs, ks, vs, diagnostic=True)

# Prepare outside capture; keep this owner alive while a graph uses its buffers.
from flag_omni.lightx2v.sage_attention_s5000 import prepare_sage_attention_s5000
prepared = prepare_sage_attention_s5000(qi, ki, vf, qs, ks, vs)
y = prepared.run()
```

Preparation allocates K/V/output storage, creates aligned device TME descriptors
and compiles the kernel. Each run packs current K into padded HND and permutes V
within each 128-key group with `f(k) = (k % 8) * 16 + k // 8`, preserving bytes.
Update inputs in place, never replace their storage while reusing a prepared
launch. Outputs are reused. Do not execute one prepared workspace concurrently
on different streams. Preparation compiles and loads both the packing and main kernels before graph capture.

### Compiler dependency

Use a compatible MP31 FlagTree build containing the qualified v153 capabilities.
The qualification used CPython 3.10, MUSA SDK 4.3.5 and the distribution
`flagtree-0.7.0+mthreads.s5000.r3473.v153.20261110` (Linux x86_64 wheel SHA256
`08d8e94ffe913e76ea5094df5ece7d5aca6afa37c0412ec62523e50ed38efc16`).
This custom compiler is an external prerequisite, not downloaded or installed
by FlagOmni. An arbitrary upstream Triton/FlagTree release is insufficient.

The `config.py` module records all exact options, register layouts and ASTSource
alignment attributes. It uses external SDK `bin/llc` / `bin/ld.lld` and the
`native_mp31_sg16` descriptor ABI; do not substitute the bundled generic tools
just because both identify as LLVM 14. `sdk_path` can select an equivalent SDK
installation. Compiler failures are surfaced, not replaced by another kernel.
The compiler and driver are not modified by this package.

## Source provenance

- H3 operators: [FlagGems-lightx2v PR #1](https://github.com/CherryLemon/FlagGems-lightx2v/pull/1),
  commit `df5f394ec0b091df9059ef05bdd601815304495a`.
- Generic Sage and formatted NV implementation: [FlagGems PR #8459](https://github.com/flagos-ai/FlagGems/pull/8459),
  commit `87d6383fc99e316768380e7a0633690510015f64`.
- NV was compared against the final frozen pipeline: wrapper/quantization/native
  mean ASTs match after import/docstring normalization. The core differs only
  by removal of an unused layout assignment and renaming an unused result.
  Frozen source SHA256: core
  `81a5489e982e1b2f39d21e44b34b75951498c292b5d0891563e4fb1d83a63b9b`,
  native mean `817be5a0f834e88cc3151ce2dc8b8420a37deabac67c4eb59a0c21e076893c14`.
- MP31 final source snapshot (2026-11-10): `sage_triton.py` SHA256
  `8a21c30982515a52df910e70448fbd79abd521b77a1aa7dd5a1fc30ab8737829`,
  layout source SHA256
  `cecfc81ae9ad363b80d83ade76cf4662dbaa21f331d582edc4c5d71f46bddd48`.
  Namespace changes, Apache headers and formatting are separated from the
  unchanged kernel/layout ASTs in the migration verification.

## Validation and benchmark scope

See [migration validation](lightx2v-validation.md) for checks on this port.
Historical evidence must not be relabeled as validation of a new packaged head.

Historical final NV 31-group full-wrapper Graph medians were 4.805240 ms (720p)
and 24.048529 ms (1080p), including quantization but not a multi-GPU model run.
The corresponding old TLE medians were 5.512200 / 26.490152 ms and native Sage
5.312064 / 25.860920 ms. These are not fresh benchmark results for this port,
and the 1080p main-kernel confidence interval did not establish a stable lead.

Historical MP31 R3473 core medians were 8.824860 / 8.825660 / 8.825060 ms.
Those exclude quantization, layout packing, descriptor creation, wrapper and
model costs. The 8 ms goal was not achieved. The later SDK14-pass result is not
claimed for this no-pass delivery. Historical full FP32 independent-reference
checks, exact same-call BF16 rounding, 2048 repeats and 54 dynamic Graph checks
apply to the frozen core, not automatically to the new Python wrapper.

```bash
# CPU contracts and GPU test collection
pytest tests/lightx2v --collect-only -q
pytest tests/lightx2v/test_sage_attention_contract.py -q

# NVIDIA numerical/reference tests (install native SageAttention for the oracle)
pytest tests/lightx2v/test_sage_attention.py --accelerator-device cuda:0 --require-accelerator -q

# MUSA numerical tests, only on exclusively allocated MP31 hardware
pytest tests/lightx2v/test_sage_attention_s5000.py --accelerator-device musa:0 --require-accelerator -q

# Optional independent full-packet oracle. Directory contains q/k/v/qs/ks/vs.bin
# and reference.fp32.bin in the documented shapes, all stored in C order.
FLAGOMNI_S5000_FIXTURES=/path/to/fixtures pytest tests/lightx2v/test_sage_attention_s5000.py \
  --accelerator-device musa:0 --require-accelerator -q

# Full-call paired native-SM90 timing, including quantization and allocation
pytest benchmark/lightx2v/test_sage_attention.py --accelerator-device cuda:0 --require-accelerator --tier sm90 -s

# Prepared S5000 timing includes current K/V packing + core, not raw quantization
FLAGOMNI_BENCH_S5000=1 pytest benchmark/lightx2v/test_sage_attention_s5000.py \
  --accelerator-device musa:0 --require-accelerator -s
```
