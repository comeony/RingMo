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
"""attention layer"""
from mindspore import nn, Tensor, context
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.transformer.op_parallel_config import default_dpmp_config

from ringmo_framework.models.core.relative_pos_bias import RelativePositionBias, RelativePositionBiasForSwin
from .layers import Linear, Dropout


class Attention(nn.transformer.transformer.MultiHeadAttention):
    r"""
        This is an implementation of multihead attention in the paper `Attention is all you need
        <https://arxiv.org/pdf/1706.03762v5.pdf>`_. Given the query vector with source length, and the
        key and value vector with target length, the attention will be performed as the following
    """

    def __init__(self,
                 batch_size,
                 src_seq_length,
                 tgt_seq_length,
                 hidden_size,
                 num_heads,
                 window_size=None,
                 hidden_dropout_rate=0.,
                 attention_dropout_rate=0.,
                 weight_init='XavierUniform',
                 compute_dtype=mstype.float16,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 parallel_config=default_dpmp_config):
        super(Attention, self).__init__(
            batch_size,
            src_seq_length,
            tgt_seq_length,
            hidden_size,
            num_heads,
            hidden_dropout_rate=hidden_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            compute_dtype=compute_dtype,
            softmax_compute_type=softmax_compute_type,
            param_init_type=param_init_type,
            parallel_config=parallel_config
        )
        self._is_ascend = context.get_context('device_target') in ["Ascend"]
        # Output layer
        self.projection = Linear(in_channels=hidden_size,
                                 out_channels=hidden_size,
                                 transpose_b=False,
                                 weight_init=weight_init,
                                 param_init_type=param_init_type).to_float(compute_dtype)
        self.projection.shard(strategy_bias=((parallel_config.data_parallel, 1), (1,)),
                              strategy_matmul=((parallel_config.data_parallel, parallel_config.model_parallel),
                                               (parallel_config.model_parallel, 1)))
        self.projection.bias.parallel_optimizer = False

        self.dropout = Dropout(1 - hidden_dropout_rate)
        self.dropout.shard(((parallel_config.data_parallel, 1),))
        self.prob_dropout = Dropout(1 - attention_dropout_rate)
        self.prob_dropout.shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))

        # Query
        self.dense1 = Linear(hidden_size,
                             hidden_size,
                             weight_init=weight_init,
                             param_init_type=param_init_type).to_float(compute_dtype)
        self.dense1.shard(strategy_matmul=((parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)),
                          strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                                         (parallel_config.model_parallel,)))
        # Key
        self.dense2 = Linear(hidden_size,
                             hidden_size,
                             weight_init=weight_init,
                             param_init_type=param_init_type).to_float(compute_dtype)
        self.dense2.shard(strategy_matmul=((parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)),
                          strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                                         (parallel_config.model_parallel,)))

        # Value
        self.dense3 = Linear(hidden_size,
                             hidden_size,
                             weight_init=weight_init,
                             param_init_type=param_init_type).to_float(compute_dtype)
        self.dense3.shard(strategy_matmul=((parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)),
                          strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                                         (parallel_config.model_parallel,)))

        if window_size:
            self.relative_position_bias = RelativePositionBias(window_size, num_heads)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None
            self.relative_position_bias = None

        self.add_3d = P.Add().shard(((parallel_config.data_parallel, 1, 1, 1), (1, 1, 1)))
        self.expand_dims_rpb = P.ExpandDims().shard(((1, 1, 1),))
        self.add_rpb = P.Add().shard(((parallel_config.data_parallel, 1, 1, 1), (1, 1, 1, 1)))

        self.softmax = nn.Softmax().to_float(softmax_compute_type)
        self.softmax.softmax.shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
        self.softmax_3d = nn.Softmax().to_float(softmax_compute_type)
        self.softmax_3d.softmax.shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1),))

    def construct(self, query_tensor, key_tensor, value_tensor, attention_mask, key_past=None,
                  value_past=None, batch_valid_length=None, rel_pos_bias=None):
        """construct of attention"""
        self._check_inputs(query_tensor, key_tensor, value_tensor, attention_mask, key_past,
                           value_past, batch_valid_length)
        query_tensor, key_tensor, value_tensor, batch_size, ori_shape = self._convert_to_2d_tensor(query_tensor,
                                                                                                   key_tensor,
                                                                                                   value_tensor,
                                                                                                   attention_mask)

        # multi head attention: query, key, value are derived from the same inputs
        query = self.dense1(query_tensor)
        key = self.dense2(key_tensor)
        value = self.dense3(value_tensor)
        # the returned shape is [bs, num_heads, seq_length, size_per_head]
        query = self.transpose(
            F.reshape(
                query,
                (batch_size, -1, self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        # the returned shape is [bs, size_per_head, seq_length, num_heads]
        key = self.transpose(
            F.reshape(
                key, (batch_size, -1, self.n_head, self.size_per_head)),
            (0, 2, 3, 1))
        # the returned shape is [bs, num_heads, seq_length, size_per_head]
        value = self.transpose(
            F.reshape(
                value,
                (batch_size, -1, self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        # support input shape is [bs, seq, seq] or [bs, heads, seq, seq]
        if len(F.shape(attention_mask)) == 3:
            # expand attention mask from [bs, seq, seq] -> [bs, 1, seq, seq]
            attention_mask = self.expand_dims(attention_mask, 1)
        # multi head attention considering attention mask
        # the return shape is [bs * seq_length, hidden_size]
        attention = self._attn(query, key, value, attention_mask, rel_pos_bias)
        # Output
        output = self.projection(attention)
        output = self.dropout(output)
        output = F.reshape(output, ori_shape)
        return output

    def _softmax(self, attention_scores):
        """
        For the consideration of the performance, do softmax according to different situations
        :param attention_scores: a 3d tensor before softmax
        :return: the attention scores.
        """

        if self._is_ascend and self.softmax_dtype == mstype.float16 or not self._is_ascend:
            attention_probs = self.softmax(attention_scores)
        else:
            shape = F.shape(attention_scores)
            # attention probs
            attention_probs = self.softmax_3d(
                F.reshape(attention_scores,
                          (shape[0], -1, shape[-1])))
            attention_probs = F.reshape(attention_probs, shape)
        return attention_probs

    def _attn(self, query, key, value, attention_mask, rel_pos_bias):
        r"""Get the weighted score along the seq_length."""
        # Normalize query and key before MatMul, default off
        # Attention score [bs, num_heads, seq_length, seq_length]
        factor = P.Cast()(self.scale_factor, P.DType()(query))
        query = self.real_div(query, factor)
        key = self.real_div(key, factor)
        score = self.batch_matmul(query, key)

        ori_dtype = P.DType()(score)
        score = P.Cast()(score, self.softmax_dtype)

        # Minus 10000 for the position where masked to exclude them from softmax
        multiplu_out = self.sub(
            P.Cast()(F.tuple_to_array((1.0,)), P.DType()(score)),
            P.Cast()(attention_mask, P.DType()(score)))

        adder = self.mul(multiplu_out, self.multiply_data)
        attention_scores = self.add(adder, score)

        # add window cross module
        if self.relative_position_bias is not None:
            relative_position_bias = self.expand_dims_rpb(self.relative_position_bias(), 0)
            attention_scores = self.add_rpb(attention_scores, relative_position_bias)

        if rel_pos_bias is not None:
            attention_scores = self.add_3d(attention_scores, rel_pos_bias)

        # attention probs
        attention_probs = self._softmax(attention_scores)
        attention_probs = P.Cast()(attention_probs, ori_dtype)

        attention_probs = self.prob_dropout(attention_probs)
        # Weighted sum output [bs, num_heads, seq_length, size_per_head]
        weighted_values = self.batch_matmul(attention_probs, value)
        attention_merge = self._merge_heads(weighted_values)
        return attention_merge


class WindowAttention(nn.Cell):
    r""" Window based multi-head self attention (W-MSA) Cell with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 compute_dtype=mstype.float16,
                 param_init_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 parallel_config=default_dpmp_config):
        super(WindowAttention, self).__init__()
        if isinstance(dim, tuple) and len(dim) == 1:
            dim = dim[0]
        if parallel_config:
            dp = parallel_config.data_parallel
            mp = parallel_config.model_parallel
        else:
            dp = mp = 1
        self._is_ascend = context.get_context('device_target') in ["Ascend"]

        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale_factor = Tensor(qk_scale or head_dim ** -0.5, mstype.float32)
        self.relative_position_bias = RelativePositionBiasForSwin(self.window_size, num_heads)

        # get pair-wise relative position index for each token inside the window
        self.q = Linear(
            in_channels=dim, out_channels=dim, has_bias=qkv_bias,
            param_init_type=param_init_type).to_float(compute_dtype)
        self.q.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))

        self.k = Linear(
            in_channels=dim, out_channels=dim, has_bias=qkv_bias,
            param_init_type=param_init_type).to_float(compute_dtype)
        self.k.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))

        self.v = Linear(
            in_channels=dim, out_channels=dim, has_bias=qkv_bias,
            param_init_type=param_init_type).to_float(compute_dtype)
        self.v.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))

        self.attn_drop = Dropout(keep_prob=1.0 - attn_drop)
        self.proj = Linear(
            in_channels=dim, out_channels=dim, has_bias=True,
            param_init_type=param_init_type).to_float(compute_dtype)
        self.proj.shard(strategy_bias=((dp, 1), (1,)), strategy_matmul=((dp, mp), (mp, 1)))

        self.proj_drop = Dropout(keep_prob=1.0 - proj_drop)
        self.proj_drop.shard(((dp, mp, 1, 1),))

        self.softmax = nn.Softmax().to_float(softmax_compute_type)
        self.softmax.softmax.shard(((dp, mp, 1, 1),))
        self.softmax_3d = nn.Softmax().to_float(softmax_compute_type)
        self.softmax_3d.softmax.shard(((dp, mp, 1),))
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(((dp, 1, 1, 1),))
        self.matmul = P.BatchMatMul().shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.real_div = P.RealDiv().shard(((dp, mp, 1, 1), ()))
        self.sub = P.Sub().shard(((1,), (dp, 1, 1, 1)))
        self.mul = P.Mul().shard(((dp, 1, 1, 1), (1,)))
        self.add_4d = P.Add().shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        self.add_5d = P.Add().shard(((dp, 1, 1, 1, 1), (dp, 1, 1, 1, 1)))
        self.dtype = P.DType()
        self.shape = P.Shape()
        self.compute_type = compute_dtype
        self.softmax_dtype = softmax_compute_type

    def _softmax(self, attention_scores):
        """
        For the consideration of the performance, do softmax according to different situations
        :param attention_scores: a 3d tensor before softmax
        :return: the attention scores.
        """

        if self._is_ascend and self.softmax_dtype == mstype.float16 or not self._is_ascend:
            attention_probs = self.softmax(attention_scores)
        else:
            shape = self.shape(attention_scores)
            # attention probs
            attention_probs = self.softmax_3d(
                self.reshape(attention_scores,
                             (shape[0], -1, shape[-1])))
            attention_probs = self.reshape(attention_probs, shape)
        return attention_probs

    def construct(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b, seq, c = x.shape
        ori_type = self.dtype(x)
        x = self.cast(x, self.compute_type)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q = self.transpose(self.reshape(q, (b, seq, self.num_heads, c // self.num_heads)), (0, 2, 1, 3))
        k = self.transpose(self.reshape(k, (b, seq, self.num_heads, c // self.num_heads)), (0, 2, 3, 1))
        v = self.transpose(self.reshape(v, (b, seq, self.num_heads, c // self.num_heads)), (0, 2, 1, 3))
        factor = self.cast(self.scale_factor, self.compute_type)
        q = self.mul(q, factor)
        attn = self.matmul(q, k)

        attn = self.cast(attn, ori_type)
        attn = self.add_4d(attn, self.relative_position_bias())

        if mask is not None:
            nw, ws2, _ = mask.shape
            # mask: [64, 49, 49] ==> [1, 64, 1, 49, 49]
            mask = self.reshape(mask, (1, nw, 1, ws2, ws2))
            attn = self.reshape(attn, (b // nw, nw, self.num_heads, seq, seq))
            attn = self.add_5d(attn, mask)
            attn = self.reshape(attn, (-1, self.num_heads, seq, seq))
            attn = self._softmax(attn)
        else:
            attn = self._softmax(attn)
        attn = self.cast(attn, self.compute_type)
        attn = self.attn_drop(attn)

        x = self.matmul(attn, v)
        x = self.transpose(x, (0, 2, 1, 3))
        x = self.reshape(x, (b, seq, c))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'
