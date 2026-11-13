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

"""Byte-preserving NHD to padded HND packing for the qualified core."""

import triton
import triton.language as tl


@triton.jit
def pack_kv(K, V, KP, VP, BLOCK: tl.constexpr):
    # V is addressed as uint8: no numeric FP8 conversion or NaN canonicalization.
    head = tl.program_id(1)
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = index // 128, index % 128
    valid = index < 34560 * 128
    source = (row * 7 + head) * 128 + col
    k = tl.load(K + source, valid & (row < 34498), 0)
    v = tl.load(V + source, valid & (row < 34498), 0)
    target_k = (head * 34560 + row) * 128 + col
    permuted = (row // 128) * 128 + (row % 8) * 16 + (row % 128) // 8
    target_v = (head * 34560 + permuted) * 128 + col
    tl.store(KP + target_k, k, valid)
    tl.store(VP + target_v, v, valid)
