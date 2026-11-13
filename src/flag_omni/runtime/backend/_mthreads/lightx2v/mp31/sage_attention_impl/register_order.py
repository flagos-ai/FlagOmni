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

from triton.experimental.tle.language.gpu.types import distributed_encoding


class RegisterOrder(distributed_encoding):
    def __init__(self, reg, lane, warp, shape):
        self.reg = tuple(map(tuple, reg))
        self.lane = tuple(map(tuple, lane))
        self.warp = tuple(map(tuple, warp))
        self.shape = tuple(shape)

    @property
    def rank(self):
        return len(self.shape)

    def to_ir(self, builder):
        return builder.get_tle_linear_encoding(
            self.reg, self.lane, self.warp, [], self.shape
        )

    def __hash__(self):
        return hash((self.reg, self.lane, self.warp, self.shape))

    def __eq__(self, x):
        return type(x) == type(self) and (  # noqa: E721 - qualified layout identity
            self.reg,
            self.lane,
            self.warp,
            self.shape,
        ) == (x.reg, x.lane, x.warp, x.shape)

    def __repr__(self):
        return "RegisterOrder" + repr((self.reg, self.lane, self.warp, self.shape))
