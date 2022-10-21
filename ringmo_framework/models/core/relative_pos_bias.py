# Copyright 2022 Huawei Technologies Co., Ltd
# Copyright 2022 Aerospace Information Research Institute,
# Chinese Academy of Sciences.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""relative pos bias of ringmo"""
import numpy as np

from mindspore import nn
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
import mindspore.common.initializer as weight_init
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P


class RelativePositionBias(nn.Cell):
    """relative position bias"""
    def __init__(self, window_size, num_heads):
        super(RelativePositionBias, self).__init__()

        self.window_size = window_size
        # cls to token & token to cls & cls to cls
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3

        self.relative_position_bias_table = Parameter(
            weight_init.initializer(
                weight_init.TruncatedNormal(sigma=.02),
                (self.num_relative_distance, num_heads)),
            name='relative_position_bias_table')

        # get pair-wise relative position index for each token inside the window
        coords_h = Tensor(np.arange(window_size[0]), mstype.int32)
        coords_w = Tensor(np.arange(window_size[1]), mstype.int32)
        coords = P.Stack(axis=0)(P.Meshgrid(indexing='ij')((coords_h, coords_w)))  # 2, Wh, Ww
        coords_flatten = P.Flatten()(coords)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = P.Transpose()(relative_coords, (1, 2, 0)).asnumpy()  # Wh*Ww, Wh*Ww, 2

        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1

        relative_position_index = \
            np.zeros(((window_size[0] * window_size[1] + 1),) * 2, dtype=int)

        relative_position_index[1:, 1:] = np.sum(relative_coords, axis=-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        relative_position_index = Tensor(relative_position_index, mstype.int32)
        self.relative_position_index = Parameter(
            relative_position_index,
            requires_grad=False, name="relative_position_index")

        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

        self.gather = P.Gather()

    def construct(self):
        relative_position_index = self.relative_position_index.view(-1)
        relative_position_bias = self.gather(self.relative_position_bias_table, relative_position_index, 0)
        relative_position_bias = self.reshape(
            relative_position_bias,
            (self.window_size[0] * self.window_size[1] + 1,
             self.window_size[0] * self.window_size[1] + 1, -1))
        relative_position_bias = self.transpose(relative_position_bias, (2, 0, 1))
        return relative_position_bias


class RelativePositionBiasForSwin(nn.Cell):
    """relative position bias for swin"""
    def __init__(self, window_size, num_heads):
        super(RelativePositionBiasForSwin, self).__init__()
        self.window_size = window_size
        # cls to token & token to cls & cls to cls
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)

        self.relative_position_bias_table = Parameter(
            weight_init.initializer(
                weight_init.TruncatedNormal(sigma=.02),
                (self.num_relative_distance, num_heads)),
            name='relative_position_bias_table')

        # get pair-wise relative position index for each token inside the window
        coords_h = Tensor(np.arange(window_size[0]), mstype.int32)
        coords_w = Tensor(np.arange(window_size[1]), mstype.int32)
        coords = P.Stack(axis=0)(P.Meshgrid(indexing='ij')((coords_h, coords_w)))  # 2, Wh, Ww
        coords_flatten = P.Flatten()(coords)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = P.Transpose()(relative_coords, (1, 2, 0)).asnumpy()  # Wh*Ww, Wh*Ww, 2

        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1

        relative_position_index = Tensor(np.sum(relative_coords, axis=-1), mstype.int32)  # Wh*Ww, Wh*Ww
        self.relative_position_index = Parameter(
            relative_position_index, requires_grad=False, name="relative_position_index")

        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.expand_dim = P.ExpandDims()
        self.gather = P.Gather()

    def construct(self):
        relative_position_index = self.relative_position_index.view(-1)
        relative_position_bias = self.gather(self.relative_position_bias_table, relative_position_index, 0)
        relative_position_bias = self.reshape(
            relative_position_bias,
            (self.window_size[0] * self.window_size[1],
             self.window_size[0] * self.window_size[1], -1))
        relative_position_bias = self.transpose(relative_position_bias, (2, 0, 1))
        relative_position_bias = self.expand_dim(relative_position_bias, 0)
        return relative_position_bias
