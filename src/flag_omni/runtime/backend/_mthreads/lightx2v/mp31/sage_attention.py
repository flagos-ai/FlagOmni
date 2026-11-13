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


def _prepare(
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
    does not quantize raw QKV. Scale geometry is validated by _validate.
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


def _run_quantized(
    query,
    key,
    value,
    query_scale,
    key_scale,
    value_scale,
    *,
    workspace=None,
    return_fp32=False,
):
    """Private MP31 implementation of the public prequantized input contract.

    A caller-owned dict keeps a single launch alive for graph capture. It is
    bound to input object/storage identity, and must not be shared
    concurrently. No global tensor or descriptor cache is introduced.
    """
    import os

    import torch

    _validate(query, key, value, query_scale, key_scale, value_scale)
    if not isinstance(return_fp32, bool):
        raise TypeError("return_fp32 must be a bool")
    if workspace is not None and not isinstance(workspace, dict):
        raise TypeError("workspace must be a caller-owned dict")
    inputs = (query, key, value, query_scale, key_scale, value_scale)
    with torch.musa.device(query.device):
        signature = (tuple((id(t), t.data_ptr()) for t in inputs), return_fp32)
        state = None if workspace is None else workspace.get("_sage_attention")
        if state is not None:
            if state[0] != signature:
                raise ValueError(
                    "SageAttention workspace is bound to its input storage"
                )
            prepared = state[1]
        else:
            if torch.musa.is_current_stream_capturing():
                raise RuntimeError(
                    "Warm up sage_attention with a workspace before graph capture"
                )
            prepared = _prepare(
                *inputs,
                diagnostic=return_fp32,
                sdk_path=os.environ.get("FLAGOMNI_MUSA_SDK", "/usr/local/musa"),
            )
            if workspace is not None:
                workspace["_sage_attention"] = (signature, prepared)
        return prepared.run()
