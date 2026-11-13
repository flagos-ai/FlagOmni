"""CUDA-compatible device names must not select NVIDIA kernels."""

import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "backend,arch,expected",
    [("cuda", 90, True), ("hip", 90, False), ("xpu", 90, False), ("cuda", 89, False)],
)
def test_sm90_requires_nvidia_compiler_target(monkeypatch, backend, arch, expected):
    path = Path(__file__).parents[2] / "src/flag_omni/runtime/capabilities.py"
    spec = importlib.util.spec_from_file_location("capabilities_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(device=lambda d: nullcontext())),
    )
    target = SimpleNamespace(backend=backend, arch=arch)
    monkeypatch.setitem(
        sys.modules,
        "triton",
        SimpleNamespace(
            runtime=SimpleNamespace(
                driver=SimpleNamespace(
                    active=SimpleNamespace(get_current_target=lambda: target)
                )
            )
        ),
    )
    assert module.is_nvidia_sm90(SimpleNamespace(type="cuda")) is expected
    assert not module.is_nvidia_sm90(SimpleNamespace(type="musa"))
