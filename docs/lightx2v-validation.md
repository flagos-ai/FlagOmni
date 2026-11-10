# LightX2V and SageAttention migration validation

Validation date: 2026-11-10. This document distinguishes checks on the migrated
package from historical prototype measurements. No new end-to-end model speedup,
A100 acceptance or cross-device numerical equivalence is claimed.

## Source preservation

- H3 operators were migrated from `df5f394ec0b091df9059ef05bdd601815304495a` in
  FlagGems-lightx2v PR #1 into framework/vendor directories. Kernel numerical
  expressions are unchanged. The former FlagGems libdevice shim import was
  replaced by Triton's portable libdevice module; both NVIDIA and MUSA compilers
  map it to the same vendor `exp`, not approximate `tl.exp`.
- Generic SageAttention and the NV formatted source are from FlagGems PR #8459
  at `87d6383fc99e316768380e7a0633690510015f64`. The NV implementation was
  independently compared with frozen final `sage_tle_pipeline_v4`.
- MP31 final kernel and register-layout ASTs exactly match the qualified R3473
  source. The later R3704/R3791 source files are identical, but their optional
  SDK14-pass compiler settings are intentionally not included.
- Source identities and transformations are in
  [the provenance manifest](lightx2v-source-provenance.json).

## Checks on this migration

| Check | Result / scope |
|---|---|
| CPU suite | 43 passed, 41 accelerator cases skipped |
| Test and benchmark collection | 90 cases collected |
| Full repository pre-commit | YAML, EOF, whitespace, flake8, isort, Black and Black-Jupyter passed |
| Packaging | wheel and source distribution built |
| H100 suite | 82 passed, 2 explicit MUSA tests skipped, including the final portable-libdevice fix |
| MP31 CPU recompilation | perf and diagnostic code, data and resource sections match qualified R3473 |
| S5000 fixed-shape suite | 13 passed: 11 CPU contracts, 2 hardware numerical/Graph cases |

H100 uses Torch 2.11.0+cu130, Triton 3.6.0 with the compatible FlagTree TLE
runtime. Tests cover H3 independent native-order references, the >INT32 SwiGLU
case, ordinary Sage quantization-aware references, SM90 native bitwise equality,
NHD/HND, partial tiles, K-mean boundaries, dynamic Graph inputs, zero V and
noncontiguous V strides. Native-reference tests use the separate SageAttention
reference package. They do not turn a microbenchmark into full-model validation.

S5000 uses Torch/torch_musa 2.9.0, qualified FlagTree v153, SDK 4.3.5 and driver
3.3.5-server. Recompiled main-kernel ELF differs only in DWARF debugging sections
after source formatting and line-map changes. All non-debug code/data sections
and resource metadata match. No SDK14 custom pass was loaded.

New S5000 wrapper coverage:

- Uniform-score attention versus an independent CPU arithmetic-mean oracle.
- All padded K and permuted V bytes, including 62 padded rows.
- All 30,910,208 FP32 outputs of the qualified real prequantized packet versus
  the independent FP32 oracle at `atol=rtol=1e-3`.
- Same-call exact BF16 RNE, perf-versus-diagnostic BF16 equality, eight repetitions.
- Captured prepared launch with in-place V-scale refresh and eager equivalence.

Raw model-derived fixtures and the large independent reference are external to
this public repository. The optional test accepts `FLAGOMNI_S5000_FIXTURES`,
whose contents are documented in the SageAttention guide. The synthetic
arithmetic-mean test needs no external fixture. The fixed-shape qualification
does not establish arbitrary shapes, raw-QKV quantization, multi-batch support
or model-level integration on S5000.

## Findings resolved during migration

The first MP31 comparator classified a DWARF `.debug_info` change as a failure;
inspection confirmed only debug sections differed. The first device invocation
needed the existing SDK libraries added to the task-local `LD_LIBRARY_PATH`.
A real new-wrapper bug omitted constexpr Python arguments required by the
compiled-kernel launcher. The wrapper was corrected and the complete 13-case
S5000 suite passed afterward. No kernel numerical expression was changed to
make these checks pass.

Static review also caught device-bound compiled-handle caching, architecture
checking before entering the input device context, lost rejection of code
replacement settings, and first-use packing compilation during capture. The
wrapper now keeps handles per device, checks the selected device, rejects LLVM/
MUBIN replacement, and prepares both kernels outside capture.

## Performance and remaining limits

The repository supplies paired full-call NV benchmarks and an explicitly scoped
prepared prequantized S5000 benchmark. These new entries were not retimed in
this migration. Historical numbers in the SageAttention guide retain their
original scope and are not presented as new-head measurements.

Wider FP16/multi-batch/long-sequence coverage, A100 execution, other chip backends,
full-resolution/model matrices and multi-GPU generation acceptance remain
separate work. A CPU CI pass is not GPU acceptance, and acceptance on one backend
does not imply another backend uses the same quantization semantics.
