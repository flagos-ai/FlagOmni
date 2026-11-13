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

"""CPU-only API checks. No device allocation or compiler imports required."""

import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

api = importlib.import_module("flag_omni.lightx2v.ops.sage_attention.api")


class TensorMetadata:
    ndim = 4
    shape = (1, 17, 2, 128)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    requires_grad = False

    def stride(self, dim=None):
        strides = (4352, 256, 128, 1)
        return strides if dim is None else strides[dim]

    def numel(self):
        return 4352


def test_sage_import_is_lazy():
    root = Path(__file__).resolve().parents[2] / "src"
    script = f"""
import sys
sys.path.insert(0, {str(root)!r})
from flag_omni.lightx2v.ops.sage_attention.api import sage_attention
assert not any("sage_attention_impl" in name for name in sys.modules)
assert not any(name == "triton" or name.startswith("triton.") for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"tensor_layout": "bad"}, ValueError),
        ({"is_causal": True}, NotImplementedError),
        ({"return_lse": True}, NotImplementedError),
        ({"sm_scale": 0}, ValueError),
        ({"sm_scale": float("nan")}, ValueError),
        ({"pv_dtype": "float32"}, NotImplementedError),
        ({"implementation": "tle"}, NotImplementedError),
        ({"implementation": "sm90"}, NotImplementedError),
        ({"implementation": "baseline", "pv_dtype": "fp8"}, NotImplementedError),
    ],
)
def test_unsupported_semantics(kwargs, error):
    q = TensorMetadata()
    with pytest.raises(error):
        api.sage_attention(q, q, q, **kwargs)


def test_cpu_rejected():
    q = torch.empty((1, 17, 2, 128), dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="CUDA or MUSA"):
        api.sage_attention(q, q, q)


def test_fp16_baseline_rejected():
    q = TensorMetadata()
    q.dtype = torch.float16
    with pytest.raises(NotImplementedError, match="BF16"):
        api.sage_attention(q, q, q)


def test_kernel_failure_is_not_caught(monkeypatch):
    q = TensorMetadata()
    monkeypatch.setitem(
        sys.modules,
        "flag_omni.lightx2v.ops.sage_attention.baseline",
        SimpleNamespace(core=object()),
    )

    def fail(*args, **kwargs):
        raise RuntimeError("sentinel execution failure")

    monkeypatch.setattr(api, "run_bf16_sage", fail)
    with pytest.raises(RuntimeError, match="sentinel execution failure"):
        api.sage_attention(q, q, q)


def test_fp8_rejects_other_architecture_before_import(monkeypatch):
    before = {name for name in sys.modules if "sage_attention_impl" in name}
    q = TensorMetadata()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    with pytest.raises(NotImplementedError, match="SM90"):
        api.sage_attention(q, q, q, pv_dtype="fp8")
    assert {name for name in sys.modules if "sage_attention_impl" in name} == before


def test_only_one_public_sage_operator():
    import flag_omni.lightx2v as ops

    assert [name for name in ops.__all__ if name.startswith("sage_attention")] == [
        "sage_attention"
    ]
    assert [
        name for name in ops.all_registered_ops() if name.startswith("sage_attention")
    ] == ["sage_attention"]
    with pytest.raises(ValueError, match="Unknown"):
        ops.get_op("sage_attention_s5000_quantized")


def test_prequantized_public_call_dispatches_without_changing_metadata(monkeypatch):
    import flag_omni.lightx2v as ops
    from flag_omni.runtime import sage_attention_dispatch

    q = TensorMetadata()
    q.device = SimpleNamespace(type="musa")
    packet = dict(
        format="per_thread_int8_fp8",
        query_scale=object(),
        key_scale=object(),
        value_scale=object(),
    )
    workspace = {}
    calls = []

    def backend(*args, **kwargs):
        calls.append((args, kwargs))
        return "output"

    def select(device):
        assert device.type == "musa"
        return backend

    monkeypatch.setattr(sage_attention_dispatch, "quantized_backend", select)
    assert (
        ops.sage_attention(
            q, q, q, pv_dtype="fp8", quantization=packet, workspace=workspace
        )
        == "output"
    )
    assert calls == [
        (
            (
                q,
                q,
                q,
                packet["query_scale"],
                packet["key_scale"],
                packet["value_scale"],
            ),
            dict(workspace=workspace, return_fp32=False),
        )
    ]


def test_prequantized_execution_failure_is_not_retried(monkeypatch):
    from flag_omni.runtime import sage_attention_dispatch

    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("prequantized execution failed")

    monkeypatch.setattr(
        sage_attention_dispatch, "quantized_backend", lambda device: fail
    )
    q = TensorMetadata()
    packet = dict(
        format="per_thread_int8_fp8", query_scale=1, key_scale=1, value_scale=1
    )
    with pytest.raises(RuntimeError, match="prequantized execution failed"):
        api.sage_attention(q, q, q, pv_dtype="fp8", quantization=packet)
    assert calls == [1]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(pv_dtype="bf16"),
        dict(pv_dtype="fp8", tensor_layout="HND"),
        dict(pv_dtype="fp8", is_causal=True),
        dict(pv_dtype="fp8", sm_scale=1.0),
    ],
)
def test_prequantized_unsupported_contract_is_rejected_before_dispatch(
    monkeypatch, kwargs
):
    from flag_omni.runtime import sage_attention_dispatch

    monkeypatch.setattr(
        sage_attention_dispatch,
        "quantized_backend",
        lambda device: pytest.fail("dispatched"),
    )
    q = TensorMetadata()
    packet = dict(
        format="per_thread_int8_fp8", query_scale=1, key_scale=1, value_scale=1
    )
    with pytest.raises((ValueError, NotImplementedError)):
        api.sage_attention(q, q, q, quantization=packet, **kwargs)


def test_prequantized_rejects_non_musa_before_import():
    from flag_omni.runtime.sage_attention_dispatch import quantized_backend

    with pytest.raises(NotImplementedError, match="MUSA MP31"):
        quantized_backend(torch.device("cuda:0"))
