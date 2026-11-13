# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pinned R3473 compilation contract; no SDK14 custom pass is loaded."""

from pathlib import Path

OPTIONS = {
    "canonicalize_lane_shuffles": True,
    "disable_max_ilp_scheduler": False,
    "enable_backend_opt": True,
    "enable_fp8_burst2": True,
    "enable_fp_fusion": False,
    "exp2_native": True,
    "exp2_pad_short": False,
    "exp2_vector_width": 4,
    "fp32_vector_width": 4,
    "fp8_downcast_burst4": True,
    "fuse_fp8_pack_pair_sums": True,
    "hoist_sqmma_invariant_barriers": True,
    "llc_options": "-mtgpu-fop-reorder-scheduling=1 "
    "-mtgpu-prera-internal-register-allocation=1 "
    "-mtgpu-enable-internal-regs-ra=1",
    "local_max_vector_width": 4,
    "maxnreg": 128,
    "num_stages": 1,
    "num_warps": 32,
    "optimize_group_sync_phases": True,
    "order_sqmma_overlap_adds": True,
    "pack_shared_subword_stores": True,
    "pack_sqmma_overlap_adds": True,
    "pipe_sqmma_read_acquire": True,
    "sqmma_group_barriers": True,
    "sqmma_group_completed_store_reuse": True,
    "sqmma_ordered_add_limit": 4,
    "sqmma_pipe_stage_reuse": True,
    "tie_signed_vector_conversions": True,
    "tie_sqmma_overlap_adds": True,
    "tme_issue_warp": False,
}


def compiler_options(sdk_path="/usr/local/musa"):
    """Return a fresh exact option set using the externally installed MUSA SDK."""
    sdk = Path(sdk_path).resolve()
    llc, lld = sdk / "bin/llc", sdk / "bin/ld.lld"
    if not llc.is_file() or not lld.is_file():
        raise RuntimeError(
            "The qualified S5000 kernel requires MUSA SDK llc and ld.lld"
        )
    return dict(
        OPTIONS,
        llc_path=str(llc),
        llc_asm_path=str(llc),
        lld_path=str(lld),
        sqmma_descriptor_abi="native_mp31_sg16",
    )


def require_source_compilation():
    """Reject binary/LLVM substitution without changing process-wide settings."""
    from triton import knobs

    musa = getattr(knobs, "musa", None)
    if musa is None:
        raise RuntimeError(
            "The final S5000 source requires the qualified MUSA compiler"
        )
    if musa.replace_llir or musa.replace_mubin:
        raise RuntimeError(
            "S5000 SageAttention does not allow LLVM or MUBIN replacement"
        )


def compile_kernel(*, diagnostic=False, sdk_path="/usr/local/musa"):
    """Compile via ASTSource, without loading or executing a GPU kernel."""
    require_source_compilation()
    import triton
    import triton.experimental.tle.language as tle
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from .core import _core
    from .register_order import RegisterOrder

    lane = [[0, 1], [0, 2], [0, 4], [1, 0], [2, 0]]
    warp = [[4, 0], [8, 0], [32, 0], [64, 0], [128, 0]]
    constants = dict(
        NQ=34498,
        NK=34498,
        H=7,
        BM=256,
        NP=34560,
        L=tle.gpu.mthreads.MusaSqmmaEncoding([3, 1], [32, 1], [32, 128, 128]),
        LD=RegisterOrder([[16, 0]], lane, warp, [256, 8]),
        LC=RegisterOrder(
            [[0, 8], [0, 16], [0, 32], [16, 0], [0, 64]], lane, warp, [256, 128]
        ),
        LP=RegisterOrder(
            [[0, 1], [0, 2], [0, 4], [0, 8], [16, 0]],
            [[0, 16], [0, 32], [0, 64], [1, 0], [2, 0]],
            warp,
            [256, 128],
        ),
        WRITE_FP32=diagnostic,
        UNROLL=2,
        DEN_EARLY=False,
    )
    signature = dict(
        Q="*i8",
        KD="*i8",
        VD="*i8",
        QS="*fp32",
        KS="*fp32",
        VS="*fp32",
        O="*bf16",
        F="*fp32",
    )
    signature.update({name: "constexpr" for name in constants})
    attributes = {
        **{(i,): [["tt.divisibility", 16]] for i in (3, 4, 5, 6, 7)},
        **{(i,): [["tt.divisibility", 64]] for i in (0, 1, 2)},
    }
    source = ASTSource(_core, signature, constexprs=constants, attrs=attributes)
    return triton.compile(
        source, target=GPUTarget("musa", 31, 32), options=compiler_options(sdk_path)
    )
