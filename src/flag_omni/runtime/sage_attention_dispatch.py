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

"""Platform selection for SageAttention; never retry after an execution error."""


def fp8_backend(device, implementation):
    if implementation not in ("auto", "sm90"):
        raise NotImplementedError("FP8 PV requires implementation auto/sm90")
    if device.type != "cuda":
        raise NotImplementedError("FP8 PV SageAttention requires CUDA SM90")
    import torch

    if torch.cuda.get_device_capability(device) != (9, 0):
        raise NotImplementedError("FP8 PV SageAttention requires CUDA SM90")
    from .backend._nvidia.lightx2v.hopper.sage_attention import sage_attention

    return sage_attention


def baseline_launch_options(device, nq, nk):
    """Keep vendor launch tuning out of the portable operator algorithm."""
    if device.type == "musa":
        from .backend._mthreads.lightx2v.sage_attention import baseline_launch_options

        return baseline_launch_options(nq, nk)
    return dict(bm=64, warps=8, ordered_pv=False)
