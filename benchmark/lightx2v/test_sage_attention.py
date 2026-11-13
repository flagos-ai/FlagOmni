# Copyright 2026, The FlagOS Contributors.
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
#

"""Full-call SM90 FP8 timing, including mean, quantization and allocation.

Run with --tier sm90. Native Sage is the same-arithmetic reference;
SDPA timings are not substituted for this numerical contract.
"""

import os

import pytest
import torch

from flag_omni.lightx2v import sage_attention


@pytest.mark.parametrize("nq,nk", [(44, 44), (1108, 1792), (9529, 9529)])
def test_sage_attention(nq, nk, accelerator, benchmark, request):
    if request.config.getoption("--tier") != "sm90":
        pytest.skip("Select --tier sm90 for the FP8 Sage benchmark")
    if accelerator.type != "cuda" or torch.cuda.get_device_capability(accelerator) != (
        9,
        0,
    ):
        pytest.skip("The native FP8 reference requires SM90")
    native = pytest.importorskip("sageattention").sageattn_qk_int8_pv_fp8_cuda_sm90
    q = torch.randn((1, nq, 2, 128), device=accelerator, dtype=torch.bfloat16)
    k, v = [
        torch.randn((1, nk, 2, 128), device=accelerator, dtype=q.dtype)
        for _ in range(2)
    ]

    def reference():
        return native(
            q,
            k,
            v,
            tensor_layout="NHD",
            is_causal=False,
            qk_quant_gran="per_thread",
            pv_accum_dtype="fp32+fp32",
        )

    def candidate():
        return sage_attention(q, k, v, pv_dtype="fp8")

    assert torch.equal(reference(), candidate())
    benchmark(reference, candidate, shape=q.shape, dtype=q.dtype)


@pytest.mark.benchmark
def test_sage_attention_prequantized_latency(accelerator):
    if accelerator.type != "musa" or os.environ.get("FLAGOMNI_BENCH_S5000") != "1":
        pytest.skip("Set FLAGOMNI_BENCH_S5000=1 on an exclusively allocated S5000")
    q = torch.zeros((1, 34498, 7, 128), device=accelerator, dtype=torch.int8)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q, dtype=torch.float8_e4m3fn)
    qs, ks, vs = [
        torch.ones(shape, device=accelerator)
        for shape in ((1, 7, 17280), (1, 7, 1080), (1, 7, 128))
    ]
    workspace = {}

    def candidate():
        return sage_attention(
            q,
            k,
            v,
            pv_dtype="fp8",
            quantization=dict(
                format="per_thread_int8_fp8",
                query_scale=qs,
                key_scale=ks,
                value_scale=vs,
            ),
            workspace=workspace,
        )

    candidate()
    for _ in range(25):
        candidate()
    torch.musa.synchronize()
    samples = []
    for _ in range(100):
        start, end = [torch.musa.Event(enable_timing=True) for _ in range(2)]
        start.record()
        candidate()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    print(
        {
            "scope": "prequantized K/V packing plus main kernel, prepared allocations/descriptors",
            "warmup": 25,
            "samples": 100,
            "median_ms": sorted(samples)[len(samples) // 2],
            "reference": "no paired baseline measured by this entrypoint",
        }
    )
