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


@triton.jit
def _residual(X, A, S, Y, N: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0)
    a = tl.load(A + i, i < N, 0).to(tl.float32)
    s = tl.load(S + i % D, i < N, 0)
    tl.store(Y + i, x + a * s, i < N)


def residual(x, a, scale, *, block=1024, warps=4, artifacts=None):
    assert (
        x.device.type in ("cuda", "musa")
        and x.dtype == torch.float32
        and a.dtype == torch.float16
        and x.shape == a.shape
    )
    assert (
        x.is_contiguous()
        and a.is_contiguous()
        and scale.is_contiguous()
        and scale.dtype == torch.float32
    )
    assert scale.shape == (x.shape[-1],)
    y = torch.empty_like(x)
    k = _residual[(triton.cdiv(x.numel(), block),)](
        x,
        a,
        scale,
        y,
        x.numel(),
        x.shape[-1],
        block,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    if artifacts is not None:
        artifacts.append(k)
    return y
