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

"""LightX2V operators; import does not install hooks or load advanced kernels."""

from ..runtime.op_registrar import all_registered_ops, get_op
from .models.h3 import (
    h3_rms_norm,
    h3_rotary_tables,
    h3_swiglu,
    h3_vae_qk_norm_rope,
    h3_vae_residual,
    h3_vae_residual_norm,
)

__all__ = [
    "all_registered_ops",
    "get_op",
    "h3_reference_blend",
    "h3_rms_norm",
    "h3_rotary_tables",
    "h3_swiglu",
    "h3_vae_qk_norm_rope",
    "h3_vae_residual",
    "h3_vae_residual_norm",
    "sage_attention",
    "sage_attention_s5000_quantized",
]


def h3_reference_blend(sample, prediction, timestep, sigmas, step_index):
    """Call the reference-blend API without importing Triton at package import."""
    if sample.device.type not in ("cuda", "musa"):
        raise NotImplementedError("Reference blend requires CUDA or MUSA")
    from .ops.reference_blend import h3_reference_blend as implementation

    return implementation(sample, prediction, timestep, sigmas, step_index)


def sage_attention(*args, **kwargs):
    """Call the SageAttention API without importing optional kernels at package import."""
    from .ops.sage_attention.api import sage_attention as implementation

    return implementation(*args, **kwargs)


def sage_attention_s5000_quantized(*args, **kwargs):
    """Call the explicit S5000 prequantized API; never selected automatically."""
    from ..runtime.backend._mthreads.lightx2v.mp31.sage_attention import (
        sage_attention_s5000_quantized as implementation,
    )

    return implementation(*args, **kwargs)
