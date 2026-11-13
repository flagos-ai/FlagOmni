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

"""CPU contract tests; these deliberately make no GPU numerical claim."""

import subprocess
import sys

import pytest
import torch

import flag_omni.lightx2v as ops


def test_import_does_not_load_advanced_compiler_or_install_hooks():
    script = """import sys, torch
before=torch.nn.functional.silu
import flag_omni.lightx2v
assert torch.nn.functional.silu is before
assert not any('experimental.gluon' in n or 'experimental.tle' in n for n in sys.modules)
assert 'flag_gems' not in sys.modules
assert 'triton' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_operator_inventory_and_unknown_lookup():
    names = ops.all_registered_ops()
    assert names == sorted(ops.__all__[2:])
    for name in names:
        assert callable(ops.get_op(name))
    with pytest.raises(ValueError, match="Unknown"):
        ops.get_op("not_an_operator")
    with pytest.raises(ValueError, match="tier"):
        ops.get_op("packed_swiglu", tier="not_a_tier")


@pytest.mark.parametrize(
    "name,args",
    [
        ("packed_swiglu", lambda: (torch.empty(1, 8),)),
        ("rms_norm", lambda: (torch.empty(1, 128), torch.empty(128))),
        (
            "scaled_residual",
            lambda: (torch.empty(1, 2048), torch.empty(1, 2048), torch.empty(2048)),
        ),
        ("residual_rms_norm", lambda: (torch.empty(1, 2048), torch.empty(2048))),
        (
            "qk_norm_rope",
            lambda: (
                torch.empty(1, 1, 1, 64),
                torch.empty(1, 1, 1, 64),
                torch.empty(1, 1, 1, 48),
                torch.empty(1, 1, 1, 48),
            ),
        ),
        (
            "reference_blend",
            lambda: (torch.empty(1), torch.empty(1), torch.empty(1), torch.ones(2), 0),
        ),
    ],
)
def test_cpu_is_rejected_without_loading_advanced_implementations(name, args):
    loaded_before = set(sys.modules)
    with pytest.raises(NotImplementedError):
        getattr(ops, name)(*args())
    assert not any(
        "runtime.backend._nvidia" in n or "runtime.backend._mthreads" in n
        for n in set(sys.modules) - loaded_before
        if n.startswith("flag_omni")
    )


def test_lookup_cannot_bypass_public_input_guards():
    fn = ops.get_op("packed_swiglu")
    with pytest.raises(NotImplementedError):
        fn(torch.empty((1, 8)))


@pytest.mark.parametrize(
    "tier,device,capability,module",
    [
        ("baseline", "cuda", (8, 0), "lightx2v.ops.rms_norm.baseline"),
        ("auto", "cuda", (8, 0), "lightx2v.ops.rms_norm.baseline"),
        ("auto", "cuda", (9, 0), "runtime.backend._nvidia.lightx2v.hopper.rms_norm"),
        ("tle", "musa", None, "runtime.backend._mthreads.lightx2v.tle"),
    ],
)
def test_framework_qualified_dispatch(monkeypatch, tier, device, capability, module):
    from types import SimpleNamespace

    from flag_omni.runtime import op_registrar

    monkeypatch.setattr(op_registrar, "is_nvidia_sm90", lambda _: capability == (9, 0))
    monkeypatch.setattr(op_registrar, "_load", lambda path, symbol: path)
    assert (
        op_registrar.resolve_kernel(
            "rms_norm", device=SimpleNamespace(type=device), tier=tier
        )
        == module
    )


def test_registry_uses_target_framework_schema():
    from pathlib import Path

    import yaml

    registry = Path(__file__).resolve().parents[2] / "conf" / "operators.yaml"
    entries = yaml.safe_load(registry.read_text())["ops"]
    h3_entries = {
        entry["id"]: entry
        for entry in entries
        if not entry["id"].startswith("sage_attention")
    }
    assert set(h3_entries) == {
        name
        for name in ops.all_registered_ops()
        if not name.startswith("sage_attention")
    }
    assert all(entry["from"] == ["lightx2v"] for entry in h3_entries.values())


@pytest.mark.parametrize("name", ["sage_attention"])
def test_sage_lookup_remains_lazy_and_does_not_inject_h3_tiers(name):
    loaded_before = set(sys.modules)
    assert ops.get_op(name) is getattr(ops, name)
    assert not any(
        "sage_attention" in module for module in set(sys.modules) - loaded_before
    )
    with pytest.raises(ValueError, match="selects its backend automatically"):
        ops.get_op(name, tier="tle")


def test_swiglu_uses_portable_exact_libdevice_without_flaggems():
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2]
        / "src/flag_omni/lightx2v/ops/fusions/swiglu.py"
    ).read_text()
    tree = ast.parse(source)
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any((node.module or "").startswith("flag_gems") for node in imports)
    assert any(
        node.module == "triton.language.extra"
        and any(alias.name == "libdevice" for alias in node.names)
        for node in imports
    )
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_swiglu"
    )
    branch = next(node for node in kernel.body if isinstance(node, ast.If))
    exact_exp = branch.body[0].value
    assert isinstance(exact_exp, ast.Call)
    assert isinstance(exact_exp.func, ast.Attribute)
    assert exact_exp.func.value.id == "libdevice" and exact_exp.func.attr == "exp"


def test_vae_module_import_does_not_require_flaggems():
    pytest.importorskip("triton")
    script = """import sys
from flag_omni.lightx2v.ops.fusions import swiglu
assert swiglu.libdevice.__name__ == 'triton.language.extra.libdevice'
assert not any(name == 'flag_gems' or name.startswith('flag_gems.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True)
