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
