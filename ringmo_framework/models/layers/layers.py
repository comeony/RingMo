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
"""layers of ringmo"""
import numpy as np

from mindspore import nn, Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops.primitive import constexpr


@constexpr
def gen_shape(x_shape, ndim):
    return (x_shape,) + (1,) * (ndim + 1)


class LayerNorm(nn.transformer.layers._LayerNorm):
    # pylint: disable=W0212
    r"""
    A self-defined layer norm operation using reduce sum and reduce mean.
    """

    def __init__(self, normalized_shape, eps=1e-6, param_init_type=mstype.float32):
        super(LayerNorm, self).__init__(
            normalized_shape,
            eps=eps,
            param_init_type=param_init_type)


class Linear(nn.transformer.layers._Linear):
    # pylint: disable=W0212
    r"""
    Linear function for RingMo.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None,
                 transpose_b=True,
                 expert_num=1,
                 outer_batch=1,
                 activation_compute_type=mstype.float16,
                 param_init_type=mstype.float32,
                 compute_dtype=mstype.float16):
        super(Linear, self).__init__(
            in_channels,
            out_channels,
            weight_init=weight_init,
            bias_init=bias_init,
            has_bias=has_bias,
            activation=activation,
            transpose_b=transpose_b,
            expert_num=expert_num,
            outer_batch=outer_batch,
            param_init_type=param_init_type,
            compute_dtype=compute_dtype)

        self.activation_compute_type = activation_compute_type

    def construct(self, x):
        """construct of layer"""
        out_shape = P.Shape()(x)[:-1] + (self.out_channels,)
        x = P.Reshape()(x, (-1, self.in_channels))
        if self.expert_flag:
            x = P.Reshape()(x, (self.outer_batch, self.expert_num, -1, self.in_channels))
        weight = self.cast(self.weight, self.dtype)
        x = self.matmul(x, weight)
        if self.has_bias:
            x = self.bias_add(x, self.cast(self.bias, self.dtype))
        if self.activation_flag:
            x = self.activation(self.cast(x, self.activation_compute_type))
            x = self.cast(x, self.dtype)
        output = P.Reshape()(x, out_shape)
        return output


class Identity(nn.Cell):

    def construct(self, x):
        return x


class Dropout(nn.transformer.layers._Dropout):
    # pylint: disable=W0212
    r"""
        A Dropout Implements with P.DropoutGenMask and  P.DropoutDoMask for context training.
    """

    def __init__(self, keep_prob=0.5, dtype=mstype.float32):
        super(Dropout, self).__init__(keep_prob=keep_prob, dtype=dtype)


class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob, ndim=1, parallel_config=None):
        # pylint: disable=W0613
        super(DropPath, self).__init__()
        self.drop = Dropout(keep_prob=1 - drop_prob)
        shape = (1,) + (1,) * (ndim + 1)
        self.ndim = ndim
        self.mask = Tensor(np.ones(shape), dtype=mstype.float32)
        self.tile = P.Tile()
        self.mul = P.Mul()

    def construct(self, x):
        if not self.training:
            return x
        shape = gen_shape(x.shape[0], self.ndim)
        mask = self.tile(self.mask, shape)
        out = self.drop(mask)
        out = self.mul(out, x)
        return out


class VitDropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob):
        super(VitDropPath, self).__init__()
        self.drop = Dropout(keep_prob=1 - drop_prob)
        self.mask = Tensor(np.ones(1,), dtype=mstype.float32)
        self.tile = P.Tile()
        self.mul = P.Mul()

    def construct(self, x):
        if not self.training:
            return x
        mask = self.tile(self.mask, (x.shape[0],) + (1,) * (x.ndim - 1))
        out = self.drop(mask)
        out = self.mul(out, x)
        return out

    def shard(self, strategy):
        self.mul.shard(strategy)
