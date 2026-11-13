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


def test_mp31_rejects_non_musa_before_loading_tle():
    with pytest.raises(ValueError, match="same MUSA"):
        _validate(*_meta_inputs())


@pytest.mark.parametrize("index", range(6))
def test_mp31_rejects_wrong_shape(index):
    inputs = _meta_inputs()
    inputs[index] = inputs[index].flatten()
    with pytest.raises(ValueError, match="Expected shape"):
        _validate(*inputs)


def test_mp31_no_sdk14_custom_pass(tmp_path):
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


def test_mp31_requires_explicit_sdk(tmp_path):
    module = importlib.import_module(
        "flag_omni.runtime.backend._mthreads.lightx2v.mp31.sage_attention_impl.config"
    )
    with pytest.raises(RuntimeError, match="MUSA SDK"):
        module.compiler_options(tmp_path)


@pytest.mark.parametrize("field", ["replace_llir", "replace_mubin"])
def test_mp31_rejects_code_substitution(monkeypatch, field):
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


def test_quantized_workspace_reuse_and_capture_guards(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from flag_omni.runtime.backend._mthreads.lightx2v.mp31 import (
        sage_attention as backend,
    )

    capture = [False]
    monkeypatch.setattr(
        torch,
        "musa",
        SimpleNamespace(
            device=lambda device: nullcontext(),
            is_current_stream_capturing=lambda: capture[0],
        ),
        raising=False,
    )
    monkeypatch.setattr(backend, "_validate", lambda *args: None)
    calls = []

    def prepare(*inputs, **kwargs):
        calls.append(inputs)
        return SimpleNamespace(run=lambda: "result")

    monkeypatch.setattr(backend, "_prepare", prepare)
    inputs = [SimpleNamespace(device="musa:0", data_ptr=lambda: 16) for _ in range(6)]
    workspace = {}
    assert backend._run_quantized(*inputs, workspace=workspace) == "result"
    capture[0] = True
    assert backend._run_quantized(*inputs, workspace=workspace) == "result"
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="Warm up"):
        backend._run_quantized(*inputs, workspace={})
    inputs[0] = SimpleNamespace(device="musa:0", data_ptr=lambda: 16)
    with pytest.raises(ValueError, match="bound"):
        backend._run_quantized(*inputs, workspace=workspace)
