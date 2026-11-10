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

"""Final MP31 prequantized wrapper timing; no quantization cost is included."""

import os

import pytest
import torch


@pytest.mark.benchmark
def test_sage_attention_s5000_wrapper_latency(accelerator):
    if accelerator.type != "musa" or os.environ.get("FLAGOMNI_BENCH_S5000") != "1":
        pytest.skip("Set FLAGOMNI_BENCH_S5000=1 on an exclusively allocated S5000")
    from flag_omni.lightx2v.sage_attention_s5000 import prepare_sage_attention_s5000

    q = torch.zeros((1, 34498, 7, 128), device=accelerator, dtype=torch.int8)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q, dtype=torch.float8_e4m3fn)
    qs, ks, vs = [
        torch.ones(shape, device=accelerator)
        for shape in ((1, 7, 17280), (1, 7, 1080), (1, 7, 128))
    ]
    prepared = prepare_sage_attention_s5000(
        q,
        k,
        v,
        qs,
        ks,
        vs,
        sdk_path=os.environ.get("FLAGOMNI_MUSA_SDK", "/usr/local/musa"),
    )
    for _ in range(25):
        prepared.run()
    torch.musa.synchronize()
    samples = []
    for _ in range(100):
        start, end = [torch.musa.Event(enable_timing=True) for _ in range(2)]
        start.record()
        prepared.run()
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
