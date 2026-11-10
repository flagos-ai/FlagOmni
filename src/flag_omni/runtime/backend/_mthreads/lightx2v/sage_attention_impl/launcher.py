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

"""Allocation-owning wrapper around the qualified ASTSource kernel."""

import ctypes
from functools import lru_cache

import torch
import triton

from .config import compile_kernel, require_source_compilation
from .packing import pack_kv


@lru_cache(maxsize=4)
def _compiled(diagnostic, sdk_path, device_index):
    # CompiledKernel owns module/function handles bound to the first device.
    # Keep a separate owner for each device, even when the compiler cache shares bytes.
    return compile_kernel(diagnostic=diagnostic, sdk_path=sdk_path)


def _descriptor(tensor, shape, block_shape, element_type):
    fill = getattr(triton.runtime.driver.active.utils, "fill_tme_descriptor", None)
    if fill is None:
        raise RuntimeError("The compiler runtime lacks MUSA TME descriptor support")
    descriptor = fill(tensor.data_ptr(), shape, block_shape, 1, element_type, 0)
    data = ctypes.string_at(descriptor.tme_desc_cpu_ptr(), 64)
    device_descriptor = torch.tensor(
        list(data), dtype=torch.uint8, device=tensor.device
    )
    if device_descriptor.data_ptr() % 64:
        raise RuntimeError("MUSA TME descriptors require 64-byte aligned allocation")
    return device_descriptor


class PreparedSageAttention:
    """Own all descriptors, input storage references and outputs across launches.

    Each run repacks current K/V bytes, so in-place input updates are observed.
    Do not replace input storage, share this mutable output workspace across
    concurrent streams, or drop this object while a captured graph uses it.
    """

    def __init__(
        self,
        query,
        key,
        value,
        query_scale,
        key_scale,
        value_scale,
        *,
        diagnostic,
        sdk_path,
    ):
        self.inputs = (query, key, value, query_scale, key_scale, value_scale)
        self.diagnostic = diagnostic
        require_source_compilation()
        with torch.musa.device(query.device):
            target = triton.runtime.driver.active.get_current_target()
            if target.backend != "musa" or target.arch != 31:
                raise NotImplementedError("The final S5000 kernel requires MUSA MP31")
            self.kernel = _compiled(diagnostic, str(sdk_path), query.device.index)
            self.packed_key = torch.empty(
                (7, 34560, 128), dtype=torch.int8, device=query.device
            )
            self.packed_value = torch.empty_like(self.packed_key, dtype=torch.uint8)
            # MUSA's i8 enum is 0 and byte-storage u8/FP8 enum is 1.
            self.query_descriptor = _descriptor(query, [34498, 7 * 128], [256, 128], 0)
            self.key_descriptor = _descriptor(
                self.packed_key, [7 * 34560, 128], [128, 128], 0
            )
            self.value_descriptor = _descriptor(
                self.packed_value, [7 * 34560, 128], [128, 128], 1
            )
            self.output = torch.empty_like(query, dtype=torch.bfloat16)
            self.fp32 = (
                torch.empty_like(query, dtype=torch.float32) if diagnostic else None
            )
            # Materialize runtime handles before a caller starts graph capture.
            self.launch = self.kernel[(135, 7, 1)]
            # The runtime launcher receives all source arguments, including
            # constexpr Python objects (they are omitted from the device ABI).
            self.constants = tuple(
                self.kernel.src.constants[(index,)]
                for index in range(8, len(self.kernel.src.fn.arg_names))
            )
            packing_kernel = pack_kv.warmup(
                key,
                value.view(torch.uint8),
                self.packed_key,
                self.packed_value,
                BLOCK=1024,
                num_warps=4,
                grid=(4320, 7),
            )
            self.pack_launch = packing_kernel[(4320, 7, 1)]

    def run(self):
        query, key, value, qs, ks, vs = self.inputs
        with torch.musa.device(query.device):
            self.pack_launch(
                key, value.view(torch.uint8), self.packed_key, self.packed_value, 1024
            )
            self.launch(
                self.query_descriptor,
                self.key_descriptor,
                self.value_descriptor,
                qs,
                ks,
                vs,
                self.output,
                self.fp32,
                *self.constants,
            )
        return (self.output, self.fp32) if self.diagnostic else self.output
