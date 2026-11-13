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

"""Explicit fixed-shape prequantized API for the final MP31 SageAttention core."""


def _validate(query, key, value, query_scale, key_scale, value_scale):
    import torch

    tensors = (query, key, value, query_scale, key_scale, value_scale)
    if any(not isinstance(x, torch.Tensor) for x in tensors):
        raise TypeError("All six inputs must be tensors")
    expected = (
        ((1, 34498, 7, 128), torch.int8),
        ((1, 34498, 7, 128), torch.int8),
        ((1, 34498, 7, 128), torch.float8_e4m3fn),
        ((1, 7, 17280), torch.float32),
        ((1, 7, 1080), torch.float32),
        ((1, 7, 128), torch.float32),
    )
    for tensor, (shape, dtype) in zip(tensors, expected):
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"Expected shape {shape} and dtype {dtype}, got {tuple(tensor.shape)} / {tensor.dtype}"
            )
        if not tensor.is_contiguous():
            raise ValueError("The qualified S5000 contract requires contiguous inputs")
        if tensor.requires_grad:
            raise NotImplementedError("S5000 SageAttention is inference-only")
    if query.device.type != "musa" or any(x.device != query.device for x in tensors):
        raise ValueError("All inputs must be on the same MUSA device")
    if any(x.data_ptr() % 16 for x in tensors):
        raise ValueError("All tensors must be at least 16-byte aligned")


def prepare_sage_attention_s5000(
    query,
    key,
    value,
    query_scale,
    key_scale,
    value_scale,
    *,
    diagnostic=False,
    sdk_path="/usr/local/musa",
):
    """Prepare a reusable launch with stable input storage.

    Inputs are already quantized with the SM90 per-thread recipe. This function
    does not quantize raw QKV. See docs/sage_attention.md for exact scale indexing.
    Preparation performs compilation, allocation and descriptor creation. Run it
    before graph capture. Inputs may subsequently be updated in place.
    """
    _validate(query, key, value, query_scale, key_scale, value_scale)
    if not isinstance(diagnostic, bool):
        raise TypeError("diagnostic must be a bool")
    from flag_omni.runtime.backend._mthreads.lightx2v.mp31.sage_attention_impl.launcher import (
        PreparedSageAttention,
    )

    return PreparedSageAttention(
        query,
        key,
        value,
        query_scale,
        key_scale,
        value_scale,
        diagnostic=diagnostic,
        sdk_path=sdk_path,
    )


def sage_attention_s5000_quantized(
    query,
    key,
    value,
    query_scale,
    key_scale,
    value_scale,
    *,
    diagnostic=False,
    sdk_path="/usr/local/musa",
):
    """Run the qualified [1,34498,7,128] INT8-QK/FP8-PV core.

    Return BF16 output, or (BF16, FP32) when diagnostic=True. Only the historical
    prequantized core is hardware-qualified; this packaging wrapper has its own
    validation scope. No SDK14 custom pass, fallback or precision change is used.
    """
    prepared = prepare_sage_attention_s5000(
        query,
        key,
        value,
        query_scale,
        key_scale,
        value_scale,
        diagnostic=diagnostic,
        sdk_path=sdk_path,
    )
    return prepared.run()
