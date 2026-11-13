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

"""Source-first MP31 Sage kernel. No LLVM/assembly/binary postprocessing.

Inputs are already quantized. K is padded HND; V is the same HND with paired
within128 key permutation f(k)=(k%8)*16+k//8. This explicit preparation is part
of the wrapper contract and excluded from core-only timings.
"""

import triton
import triton.experimental.tle.language as tle
import triton.language as tl


@triton.jit
def tree16_per_lane(p, BM: tl.constexpr):
    # The native M32 fragment owns k=lane%8+8*j, j=0..15. Pair adjacent j,
    # then four-key/eight-key/sixteen-key sums. No cross-lane reduction here.
    a, b = tl.split(p.reshape(BM, 8, 2, 8).permute(0, 1, 3, 2))
    x = a + b
    a, b = tl.split(x.reshape(BM, 4, 2, 8).permute(0, 1, 3, 2))
    x = a + b
    a, b = tl.split(x.reshape(BM, 2, 2, 8).permute(0, 1, 3, 2))
    x = a + b
    a, b = tl.split(x.permute(0, 2, 1))
    return a + b


@triton.jit
def _core(
    Q,
    KD,
    VD,
    QS,
    KS,
    VS,
    O,
    F,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    H: tl.constexpr,
    BM: tl.constexpr,
    NP: tl.constexpr,
    L: tl.constexpr,
    LD: tl.constexpr,
    LC: tl.constexpr,
    LP: tl.constexpr,
    WRITE_FP32: tl.constexpr,
    UNROLL: tl.constexpr,
    DEN_EARLY: tl.constexpr,
):
    BN: tl.constexpr = 128
    tl.static_assert(not DEN_EARLY)
    SM: tl.constexpr = 0.127517431974411
    OFF: tl.constexpr = -8.807000160217285
    K = tle.gpu.mthreads.device_tensor_descriptor(
        KD, [NP * H, 128], [128, 1], [128, 128], tl.int8
    )
    V = tle.gpu.mthreads.device_tensor_descriptor(
        VD, [NP * H, 128], [128, 1], [128, 128], tl.float8e4nv
    )
    bh = tl.program_id(1)
    QT = tle.gpu.mthreads.device_tensor_descriptor(
        Q, [NQ, H * 128], [H * 128, 1], [BM, 128], tl.int8
    )
    m = tl.program_id(0) * BM + tle.gpu.set_layout(
        tl.arange(0, BM), tle.gpu.SlicedEncoding(1, L)
    )
    n = tle.gpu.set_layout(tl.arange(0, BN), tle.gpu.SlicedEncoding(0, L))
    d = tle.gpu.set_layout(tl.arange(0, 128), tle.gpu.SlicedEncoding(0, L))
    shared: tl.constexpr = tle.gpu.swizzled_shared_layout(
        16, 2, 16, [1, 0], [1, 1], [1, 1], [1, 1], [1, 0]
    )
    q = tle.gpu.alloc((BM, 128), tl.int8, layout=shared)
    ps = tle.gpu.alloc((BM, BN), tl.float8e4nv, layout=shared)
    shared3: tl.constexpr = tle.gpu.swizzled_shared_layout(
        16, 2, 16, [2, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 1, 0]
    )
    kk = tle.gpu.alloc((2, BN, 128), tl.int8, layout=shared3)
    vv = tle.gpu.alloc((2, BN, 128), tl.float8e4nv, layout=shared3)
    qcomplete = tle.gpu.alloc_barrier(expect_bytes=BM * 128)
    pipe = tle.pipe(capacity=2, k=kk, v=vv)
    writer = pipe.writer()
    reader = pipe.reader()
    tle.gpu.copy(QT, q, (BM, 128), (tl.program_id(0) * BM, bh * 128), barrier=qcomplete)
    first_slot = writer.acquire(0)
    tle.gpu.copy(K, first_slot.k, (BN, 128), (bh * NP, 0))
    tle.gpu.copy(V, first_slot.v, (BN, 128), (bh * NP, 0))
    writer.commit(0)
    tle.gpu.barrier_wait(qcomplete, phaseIdx=0)
    qs = tl.load(QS + bh * (tl.cdiv(NQ, 128) * 64) + (m // 16) * 8 + m % 8, m < NQ, 0)
    maximum = tle.gpu.set_layout(
        tl.full((BM,), -4999996.0, tl.float32), tle.gpu.SlicedEncoding(1, LD)
    )
    den = tle.gpu.set_layout(tl.full((BM, 8), 1.0, tl.float32), LD)
    acc = tle.gpu.set_layout(tl.zeros((BM, 128), tl.float32), L)
    izero = tle.gpu.set_layout(tl.zeros((BM, BN), tl.int32), L)
    fzero = tle.gpu.set_layout(tl.zeros((BM, 128), tl.float32), L)
    for part in tl.static_range(3):
        begin = 0 if part == 0 else (1 if part == 1 else NK // BN)
        end = 1 if part == 0 else (NK // BN if part == 1 else tl.cdiv(NK, BN))
        for block in tl.range(
            begin, end, loop_unroll_factor=(UNROLL if part == 1 else 1)
        ):
            ready = reader.wait(block)
            if block + 1 < tl.cdiv(NK, BN):
                next_slot = writer.acquire(block + 1)
                tle.gpu.copy(K, next_slot.k, (BN, 128), (bh * NP + (block + 1) * BN, 0))
                tle.gpu.copy(V, next_slot.v, (BN, 128), (bh * NP + (block + 1) * BN, 0))
                writer.commit(block + 1)
            pending = tle.gpu.wgmma(
                q, ready.slot.k, acc=izero, trans_b=True, out_dtype=tl.int32
            )
            raw = tle.gpu.set_layout(tle.gpu.wgmma_wait(0, pending).to(tl.float32), LC)
            ksc = tl.load(KS + bh * (tl.cdiv(NK, 128) * 4) + block * 4 + (n % 8) // 2)
            scalec = tle.gpu.set_layout(qs[:, None] * ksc[None, :], LC)
            if part < 2:
                score = raw
                scalec = scalec * SM
            else:
                score = tl.where(
                    block * BN + n[None, :] < NK, raw * scalec, float("-inf")
                )
                scalec = tl.full((BM, BN), SM, tl.float32)
            # Each lane owns sixteen keys with the same positive K scale.
            # Max and positive-scale fused affine transform commute, so the
            # two-lane group maximum need not be broadcast through a CTA tile.
            lane_max = tl.max(score.reshape(BM, 16, 8), 1)
            lane_scale = tl.max(scalec.reshape(BM, 16, 8), 1)
            ix = tl.broadcast_to(tl.arange(0, 8)[None, :], (BM, 8))
            pair_max = tl.maximum(lane_max, tl.gather(lane_max, ix ^ 1, axis=1))
            mx = tl.fma(pair_max, lane_scale, OFF)
            mx = tl.maximum(mx, tl.gather(mx, ix ^ 2, axis=1))
            mx = tl.maximum(mx, tl.gather(mx, ix ^ 4, axis=1))
            mx = tl.max_constancy(mx, [1, 8])
            nm = tle.gpu.set_layout(
                tl.maximum(maximum, tl.max(mx, 1)), tle.gpu.SlicedEncoding(1, LD)
            )
            p = tle.gpu.set_layout(
                tl.exp2(tl.fma(score, scalec, nm[:, None] * (-1.0))), LC
            )
            if part == 2:
                p = tl.where(block * BN + n[None, :] < NK, p, 0.0)
            # Match P columns with the pre-permuted V rows. No query shuffle.
            perm = p.reshape(BM, 16, 8).permute(0, 2, 1).reshape(BM, BN)
            ps.store(tle.gpu.set_layout(perm.to(tl.float8e4nv), LP))
            pending_pv = tle.gpu.wgmma(
                ps, ready.slot.v, acc=fzero, out_dtype=tl.float32
            )
            alpha = tl.exp2(maximum - nm)
            den = tle.gpu.set_layout(
                tl.fma(den, alpha[:, None], tree16_per_lane(p, BM)), LD
            )
            tmp = tle.gpu.wgmma_wait(0, pending_pv)
            acc = tl.fma(acc, alpha[:, None], tmp)
            reader.release(block)
            maximum = nm
    total = tl.sum(den, 1)
    vs = tl.load(VS + bh * 128 + d)
    value = (acc * (1.0 / total[:, None])) * vs[None, :]
    offsets = (m[:, None] * H + bh) * 128 + d[None, :]
    tl.store(O + offsets, value, m[:, None] < NQ)
    if WRITE_FP32:
        tl.store(F + offsets, value, m[:, None] < NQ)
