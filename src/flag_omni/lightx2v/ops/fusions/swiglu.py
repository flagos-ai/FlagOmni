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

"""Basic Triton VAE fusions. Native FP32/FP16 boundaries retained."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _swiglu(
    P, Y, N: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr, EXP: tl.constexpr
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row = i // D
    col = i % D
    x = tl.load(P + row * (2 * D) + col, i < N, 0).to(tl.float32)
    g = tl.load(P + row * (2 * D) + D + col, i < N, 0).to(tl.float32)
    if EXP == 0:
        z = libdevice.exp(-g)
    else:
        z = tl.exp(-g)
    # SiLU is a separate native FP16/BF16 tensor before the product.
    act = tl.div_rn(g, 1.0 + z).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + i, x * act, i < N)


def swiglu(p, *, block=1024, warps=4, exp=0, artifacts=None):
    assert p.device.type in ("cuda", "musa") and p.is_contiguous() and p.ndim >= 2
    assert p.dtype in (torch.float16, torch.bfloat16) and p.shape[-1] % 2 == 0
    d = p.shape[-1] // 2
    out = torch.empty((*p.shape[:-1], d), dtype=p.dtype, device=p.device)
    if out.numel():
        k = _swiglu[(triton.cdiv(out.numel(), block),)](
            p, out, out.numel(), d, block, exp, num_warps=warps, enable_fp_fusion=False
        )
        if artifacts is not None:
            artifacts.append(k)
    return out
