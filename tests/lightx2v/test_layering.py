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

"""Model adapters, portable operators, and vendor implementations stay separate."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from flag_omni import lightx2v
from flag_omni.lightx2v.models import h3
from flag_omni.runtime.sage_attention_dispatch import (
    baseline_launch_options,
    fp8_backend,
)

ROOT = Path(__file__).resolve().parents[2] / "src/flag_omni"


def test_model_contracts_keep_public_api_identity():
    for name in ("h3_rms_norm", "h3_swiglu", "h3_vae_residual", "h3_rotary_tables"):
        assert getattr(lightx2v, name) is getattr(h3, name)


def test_portable_operators_do_not_import_vendor_implementations():
    for source in (ROOT / "lightx2v/ops").rglob("*.py"):
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "runtime.backend" not in (node.module or ""), source
            elif isinstance(node, ast.Import):
                assert all("runtime.backend" not in a.name for a in node.names), source
    assert not (ROOT / "lightx2v/sage_attention_s5000.py").exists()
    assert not (ROOT / "lightx2v/h3.py").exists()
    assert (
        ROOT / "runtime/backend/_mthreads/lightx2v/mp31/sage_attention.py"
    ).is_file()


@pytest.mark.parametrize(
    "device,nq,nk,expected",
    [
        ("cuda", 4096, 128, dict(bm=64, warps=8, ordered_pv=False)),
        ("musa", 4096, 128, dict(bm=16, warps=8, ordered_pv=True)),
        ("musa", 4096, 256, dict(bm=128, warps=16, ordered_pv=False)),
        ("musa", 1024, 256, dict(bm=64, warps=8, ordered_pv=False)),
    ],
)
def test_baseline_launch_policy_preserved(device, nq, nk, expected):
    assert baseline_launch_options(SimpleNamespace(type=device), nq, nk) == expected


def test_fp8_device_rejection_precedes_backend_import(monkeypatch):
    import sys

    before = set(sys.modules)
    with pytest.raises(NotImplementedError, match="SM90"):
        fp8_backend(SimpleNamespace(type="musa"), "auto")
    assert not any("_nvidia" in name for name in set(sys.modules) - before)


def test_fp8_dispatch_returns_backend_without_running_it(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    fn = fp8_backend(SimpleNamespace(type="cuda"), "sm90")
    assert (
        fn.__module__
        == "flag_omni.runtime.backend._nvidia.lightx2v.hopper.sage_attention"
    )
