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

"""Contract tests and separately opted-in MP31 fixed-shape numerical gates."""

import importlib
import os
from pathlib import Path

import pytest
import torch

from flag_omni.runtime.backend._mthreads.lightx2v.mp31.sage_attention import _validate


def _meta_inputs():
    return [
        torch.empty((1, 34498, 7, 128), dtype=dtype, device="meta")
        for dtype in (torch.int8, torch.int8, torch.float8_e4m3fn)
    ] + [
        torch.empty(shape, dtype=torch.float32, device="meta")
        for shape in ((1, 7, 17280), (1, 7, 1080), (1, 7, 128))
    ]


def test_s5000_rejects_non_musa_before_loading_tle():
    with pytest.raises(ValueError, match="same MUSA"):
        _validate(*_meta_inputs())


@pytest.mark.parametrize("index", range(6))
def test_s5000_rejects_wrong_shape(index):
    inputs = _meta_inputs()
    inputs[index] = inputs[index].flatten()
    with pytest.raises(ValueError, match="Expected shape"):
        _validate(*inputs)


def test_s5000_no_sdk14_custom_pass(tmp_path):
    module = importlib.import_module(
        "flag_omni.runtime.backend._mthreads.lightx2v.mp31.sage_attention_impl.config"
    )
    (tmp_path / "bin").mkdir()
    for name in ("llc", "ld.lld"):
        (tmp_path / "bin" / name).touch()
    options = module.compiler_options(tmp_path)
    assert options["sqmma_descriptor_abi"] == "native_mp31_sg16"
    assert options["enable_fp_fusion"] is False
    assert options["num_warps"] == 32
    assert options["llc_path"] == str(tmp_path / "bin/llc")
    assert "-load" not in options["llc_options"]
    assert "sage-native" not in options["llc_options"]
    assert "misched=" not in options["llc_options"]
    options["num_warps"] = 1
    assert module.compiler_options(tmp_path)["num_warps"] == 32


def test_s5000_requires_explicit_sdk(tmp_path):
    module = importlib.import_module(
        "flag_omni.runtime.backend._mthreads.lightx2v.mp31.sage_attention_impl.config"
    )
    with pytest.raises(RuntimeError, match="MUSA SDK"):
        module.compiler_options(tmp_path)


@pytest.mark.parametrize("field", ["replace_llir", "replace_mubin"])
def test_s5000_rejects_code_substitution(monkeypatch, field):
    import sys
    from types import SimpleNamespace

    config = importlib.import_module(
        "flag_omni.runtime.backend._mthreads.lightx2v.mp31.sage_attention_impl.config"
    )
    settings = SimpleNamespace(replace_llir=False, replace_mubin=False)
    setattr(settings, field, "/unexpected/replacement")
    monkeypatch.setitem(
        sys.modules, "triton", SimpleNamespace(knobs=SimpleNamespace(musa=settings))
    )
    with pytest.raises(RuntimeError, match="replacement"):
        config.require_source_compilation()


@pytest.fixture
def s5000(accelerator):
    if accelerator.type != "musa":
        pytest.skip("Fixed-shape final kernel requires MUSA MP31")
    import triton

    if triton.runtime.driver.active.get_current_target().arch != 31:
        pytest.skip("Fixed-shape final kernel requires MP31")
    return accelerator


def _prepare(*inputs, diagnostic):
    from flag_omni.runtime.backend._mthreads.lightx2v.mp31.sage_attention import (
        prepare_sage_attention_s5000,
    )

    return prepare_sage_attention_s5000(
        *inputs,
        diagnostic=diagnostic,
        sdk_path=os.environ.get("FLAGOMNI_MUSA_SDK", "/usr/local/musa"),
    )


def _packing_reference(tensor):
    reference = torch.zeros((7, 34560, 128), dtype=tensor.dtype)
    reference[:, :34498] = tensor[0].permute(1, 0, 2)
    return reference


def test_s5000_uniform_attention_and_packing(s5000):
    query = torch.zeros((1, 34498, 7, 128), dtype=torch.int8, device=s5000)
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
    value = value_cpu.to(s5000)
    scales = [
        torch.ones(shape, dtype=torch.float32, device=s5000)
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
    assert torch.equal(prepared.packed_key.cpu(), _packing_reference(key.cpu()))
    packed = _packing_reference(value_cpu.view(torch.uint8))
    permutation = (
        (torch.arange(34560) // 128) * 128
        + (torch.arange(34560) % 8) * 16
        + (torch.arange(34560) % 128) // 8
    )
    expected = torch.empty_like(packed)
    expected[:, permutation] = packed
    assert torch.equal(prepared.packed_value.cpu(), expected)


def test_s5000_external_quantized_packet(s5000):
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
            torch.from_numpy(array).view(torch_dtype).reshape(shape).to(s5000)
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
