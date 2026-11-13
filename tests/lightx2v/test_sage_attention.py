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

"""Quantization-aware references; dense SDPA is not an FP8 Sage oracle."""

import os
from pathlib import Path

import pytest
import torch

from flag_omni.lightx2v import sage_attention


def quantized_reference(q, k, v):
    """Independent CPU per-thread INT8 QK / BF16 PV, one128-key tile."""
    q, k, v = [x.detach().cpu().permute(0, 2, 1, 3) for x in (q, k, v)]
    k = (k.float() - k.float().mean(2, keepdim=True).to(k.dtype).float()).to(k.dtype)
    nq, nk = q.shape[2], k.shape[2]
    qf, kf = q.float(), k.float()
    qs = torch.empty(q.shape[:-1])
    ks = torch.empty(k.shape[:-1])
    for start in range(0, nq, 64):
        for group in range(32):
            rows = [start + group // 8 * 16 + group % 8 + off for off in (0, 8)]
            rows = [r for r in rows if r < nq]
            if rows:
                scale = qf[:, :, rows].abs().amax((-2, -1)) / 127.0 + 1e-7
                qs[:, :, rows] = scale[..., None]
    for group in range(4):
        rows = [r for r in range(nk) if (r % 8) // 2 == group]
        if rows:
            scale = kf[:, :, rows].abs().amax((-2, -1)) / 127.0 + 1e-7
            ks[:, :, rows] = scale[..., None]
    qscaled, kscaled = qf / qs[..., None], kf / ks[..., None]
    qi = torch.trunc(qscaled + 0.5 * torch.where(qscaled >= 0, 1.0, -1.0))
    ki = torch.trunc(kscaled + 0.5 * torch.where(kscaled >= 0, 1.0, -1.0))
    score = (qi @ ki.transpose(-1, -2)) * (qs[..., None] * ks[..., None, :])
    score = score * (128**-0.5)
    # Probabilities are rounded before PV, but the denominator stays FP32.
    p = torch.exp(score - score.amax(-1, keepdim=True))
    out = (p.bfloat16().float() @ v.float()) / p.sum(-1, keepdim=True)
    return out.to(q.dtype).permute(0, 2, 1, 3)


@pytest.mark.parametrize("nq,nk", [(1, 1), (17, 31), (63, 127), (65, 128)])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_sage_bf16_quantized_reference(nq, nk, layout, accelerator):
    torch.manual_seed(123)
    q = torch.randn((1, nq, 2, 128), device=accelerator, dtype=torch.bfloat16)
    k, v = [
        torch.randn((1, nk, 2, 128), device=q.device, dtype=q.dtype) for _ in range(2)
    ]
    reference = quantized_reference(q, k, v)
    before = [x.clone() for x in (q, k, v)]
    inputs = [x.permute(0, 2, 1, 3) if layout == "HND" else x for x in (q, k, v)]
    actual = sage_attention(*inputs, tensor_layout=layout)
    if layout == "HND":
        actual = actual.permute(0, 2, 1, 3)
    torch.testing.assert_close(actual.cpu(), reference, atol=1e-3, rtol=1e-3)
    for original, saved in zip((q, k, v), before):
        assert torch.equal(original, saved)


@pytest.mark.parametrize("nq,nk", [(44, 44), (1108, 1792), (129, 257)])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_sage_sm90_matches_native(nq, nk, layout):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
        9,
        0,
    ):
        pytest.skip("Native FP8 Sage reference requires SM90")
    native = pytest.importorskip("sageattention").sageattn_qk_int8_pv_fp8_cuda_sm90
    torch.manual_seed(456)
    q = torch.randn((1, nq, 2, 128), device="cuda", dtype=torch.bfloat16)
    k, v = [
        torch.randn((1, nk, 2, 128), device=q.device, dtype=q.dtype) for _ in range(2)
    ]
    inputs = [x.permute(0, 2, 1, 3) if layout == "HND" else x for x in (q, k, v)]
    expected = native(
        *inputs,
        tensor_layout=layout,
        is_causal=False,
        pv_accum_dtype="fp32+fp32",
        qk_quant_gran="per_thread",
    )
    actual = sage_attention(*inputs, tensor_layout=layout, pv_dtype="fp8")
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("n", [44, 257])
def test_sage_graph_refresh(n):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
        9,
        0,
    ):
        pytest.skip("This test uses the CUDA Graph API")
    native = pytest.importorskip("sageattention").sageattn_qk_int8_pv_fp8_cuda_sm90
    inputs = [
        torch.randn((1, n, 2, 128), device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    sage_attention(*inputs, pv_dtype="fp8")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = sage_attention(*inputs, pv_dtype="fp8")
    for _ in range(3):
        for x in inputs:
            x.normal_()
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        expected = native(
            *inputs,
            tensor_layout="NHD",
            is_causal=False,
            pv_accum_dtype="fp32+fp32",
            qk_quant_gran="per_thread",
        )
        assert torch.equal(output, expected)


def test_sage_unsupported_semantics_rejected(accelerator):
    q = torch.empty((1, 1, 2, 128), device=accelerator, dtype=torch.bfloat16)
    for kwargs in (
        {"is_causal": True},
        {"return_lse": True},
        {"tensor_layout": "bad"},
        {"sm_scale": 0},
    ):
        with pytest.raises((ValueError, NotImplementedError)):
            sage_attention(q, q, q, **kwargs)


@pytest.mark.parametrize("nk", [17, 128, 129, 257])
def test_sage_sm90_zero_v(nk):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Requires SM90")
    q = torch.randn((1, 17, 2, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, nk, 2, 128), device=q.device, dtype=q.dtype)
    v = torch.zeros_like(k)
    # Native Sage can produce NaNs for zero V. The oracle here is analytical.
    result = sage_attention(q, k, v, pv_dtype="fp8")
    assert torch.equal(result, torch.zeros_like(result))


@pytest.mark.parametrize("nk", [64, 128, 129, 257])
def test_sage_sm90_strided_v(nk):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Requires SM90")
    native = pytest.importorskip("sageattention").sageattn_qk_int8_pv_fp8_cuda_sm90
    q = torch.randn((1, 17, 3, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, nk, 3, 128), device=q.device, dtype=q.dtype)
    backing = torch.randn((1, nk * 2, 3, 132), device=q.device, dtype=q.dtype)
    v = backing[:, ::2, :, :128]
    # The native transpose kernel has an alignment restriction on this layout.
    expected = native(
        q,
        k,
        v.contiguous(),
        tensor_layout="NHD",
        is_causal=False,
        qk_quant_gran="per_thread",
        pv_accum_dtype="fp32+fp32",
    )
    for _ in range(3):
        actual = sage_attention(q, k, v, pv_dtype="fp8")
        assert torch.equal(actual, expected)


@pytest.fixture
def prequantized_accelerator(accelerator):
    if accelerator.type != "musa":
        pytest.skip("Fixed-shape final kernel requires MUSA MP31")
    import triton

    if triton.runtime.driver.active.get_current_target().arch != 31:
        pytest.skip("Fixed-shape final kernel requires MP31")
    return accelerator


class _PreparedQuantizedCase:
    """Test helper exercising only the public operator, including graph replay."""

    def __init__(self, inputs, diagnostic):
        self.inputs = inputs
        self.diagnostic = diagnostic
        self.workspace = {}
        self.run()  # Create backend-owned resources before graph capture.

    def run(self):
        q, k, v, qs, ks, vs = self.inputs
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
            workspace=self.workspace,
            return_fp32=self.diagnostic,
        )


def _prepare(*inputs, diagnostic):
    return _PreparedQuantizedCase(inputs, diagnostic)


def _packing_reference(tensor):
    reference = torch.zeros((7, 34560, 128), dtype=tensor.dtype)
    reference[:, :34498] = tensor[0].permute(1, 0, 2)
    return reference


def test_prequantized_accelerator_uniform_attention(prequantized_accelerator):
    query = torch.zeros(
        (1, 34498, 7, 128), dtype=torch.int8, device=prequantized_accelerator
    )
    key = torch.zeros_like(query)
    # V varies with key, head, and channel. The zero scores have an independent
    # arithmetic-mean oracle. Explicit byte checks also verify the V permutation.
    row = torch.arange(34498)[:, None, None]
    head = torch.arange(7)[None, :, None]
    col = torch.arange(128)[None, None, :]
    value_cpu = (
        (((row + 3 * head + col) % 17 - 8).float() / 8)
        .to(torch.float8_e4m3fn)
        .unsqueeze(0)
    )
    value = value_cpu.to(prequantized_accelerator)
    scales = [
        torch.ones(shape, dtype=torch.float32, device=prequantized_accelerator)
        for shape in ((1, 7, 17280), (1, 7, 1080), (1, 7, 128))
    ]
    prepared = _prepare(query, key, value, *scales, diagnostic=True)
    output, fp32 = prepared.run()
    torch.musa.synchronize()
    reference = (
        value_cpu.float().double().mean(1, keepdim=True).float().expand_as(fp32.cpu())
    )
    torch.testing.assert_close(fp32.cpu(), reference, atol=1e-3, rtol=1e-3)
    assert torch.equal(output.cpu(), fp32.cpu().bfloat16())


def test_prequantized_accelerator_external_quantized_packet(prequantized_accelerator):
    """Optional full 30,910,208-element gate against an independent FP32 file.

    Supply q/k/v/qs/ks/vs.bin and reference.fp32.bin in the fixture directory.
    This does not bundle private model inputs in the public repository.
    """
    directory = os.environ.get("FLAGOMNI_S5000_FIXTURES")
    if not directory:
        pytest.skip("Set FLAGOMNI_S5000_FIXTURES for the independent full-output gate")
    import numpy as np

    directory = Path(directory)
    inputs = []
    specs = [
        ("q", np.int8, torch.int8, (1, 34498, 7, 128)),
        ("k", np.int8, torch.int8, (1, 34498, 7, 128)),
        ("v", np.uint8, torch.float8_e4m3fn, (1, 34498, 7, 128)),
        ("qs", np.float32, torch.float32, (1, 7, 17280)),
        ("ks", np.float32, torch.float32, (1, 7, 1080)),
        ("vs", np.float32, torch.float32, (1, 7, 128)),
    ]
    for name, dtype, torch_dtype, shape in specs:
        array = np.fromfile(directory / (name + ".bin"), dtype=dtype)
        inputs.append(
            torch.from_numpy(array)
            .view(torch_dtype)
            .reshape(shape)
            .to(prequantized_accelerator)
        )
    reference = torch.from_numpy(
        np.fromfile(directory / "reference.fp32.bin", dtype=np.float32)
    ).reshape(1, 34498, 7, 128)
    diagnostic = _prepare(*inputs, diagnostic=True)
    perf = _prepare(*inputs, diagnostic=False)
    for _ in range(8):
        output, fp32 = diagnostic.run()
        actual = perf.run()
        torch.musa.synchronize()
        torch.testing.assert_close(fp32.cpu(), reference, atol=1e-3, rtol=1e-3)
        assert torch.equal(output.cpu(), fp32.cpu().bfloat16())
        assert torch.equal(output, actual)
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        graph_output, graph_fp32 = diagnostic.run()
    for multiplier in (1.0, -1.0, 0.5, 1.0):
        # Changing VS is linear; it also tests graph scale-input refresh.
        inputs[5].mul_(multiplier)
        expected = diagnostic.run()[0].clone()
        graph.replay()
        torch.musa.synchronize()
        assert torch.equal(graph_output, expected)
        assert torch.equal(graph_output, graph_fp32.bfloat16())
