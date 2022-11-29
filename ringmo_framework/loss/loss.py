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
"""loss functions"""
import os

from mindspore import nn
from mindspore import ops as P
from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.nn.loss.loss import LossBase
from mindspore.ops import functional as F


class L1Loss(nn.Cell):
    """L1 loss"""
    def __init__(self, reduction='mean', parallel_config=None):
        super(L1Loss, self).__init__()

        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1

        self.abs = P.Abs().shard(((dp, 1, 1, 1),))
        self.sub = P.Sub().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))

        self.mul = P.Mul().shard(((), (dp, 1, 1, 1)))
        self.reduce_mean = P.ReduceMean().shard(((dp, 1, 1, 1),))
        self.reduce_sum = P.ReduceSum().shard(((dp, 1, 1, 1),))
        self.cast = P.Cast()

        self.average = True
        self.reduce = True
        if reduction == 'sum':
            self.average = False
        if reduction == 'none':
            self.reduce = False

        self.is_data_parallel = context.get_auto_parallel_context(
            "parallel_mode") == context.ParallelMode.DATA_PARALLEL
        if self.is_data_parallel:
            self.allreduce = P.AllReduce()
            self.device_num = int(os.getenv("RANK_SIZE", "1"))

    def get_axis(self, x):
        shape = F.shape(x)
        length = F.tuple_len(shape)
        perm = F.make_range(0, length)
        return perm

    def get_loss(self, x, weights=1.0):
        """get loss of L1loss"""
        input_dtype = x.dtype
        x = self.cast(x, mstype.float32)
        weights = self.cast(weights, mstype.float32)
        x = self.mul(weights, x)
        if self.reduce and self.average:
            x = self.reduce_mean(x, self.get_axis(x))
        if self.reduce and not self.average:
            x = self.reduce_sum(x, self.get_axis(x))
        x = self.cast(x, input_dtype)
        return x

    def construct(self, logits, labels):
        """construct of L1loss"""
        x_sub = self.sub(logits, labels)
        x = self.abs(x_sub)
        x = self.get_loss(x)
        if self.is_data_parallel:
            x = self.allreduce(x) / self.device_num
        return x


class MSELoss(nn.Cell):
    """MSE Loss"""
    def __init__(self, parallel_config, norm_pixel_loss=True):
        super(MSELoss, self).__init__()
        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1
        self.add_loss = P.Add().shard(((dp, 1, 1), ()))
        self.sub = P.Sub().shard(((dp, 1, 1), (dp, 1, 1)))
        self.divide = P.RealDiv().shard(((dp, 1, 1), (dp, 1, 1)))
        self.pow = P.Pow().shard(((dp, 1, 1), ()))
        self.divide1 = P.RealDiv().shard(((), ()))
        self.divide2 = P.RealDiv().shard(((dp, 1, 1), ()))
        self.square = P.Square().shard(((dp, 1, 1),))
        self.cast = P.Cast()
        self.mean1 = P.ReduceMean(keep_dims=True).shard(((dp, 1, 1),))
        self.mean2 = P.ReduceMean().shard(((dp, 1, 1),))
        self.mul = P.Mul().shard(((dp, 1), (dp, 1)))
        self.sum = P.ReduceSum().shard(((dp, 1,),))
        self.sum2 = P.ReduceSum(keep_dims=True).shard(((dp, 1, 1),))
        self.norm_pixel_loss = norm_pixel_loss

        self.is_data_parallel = context.get_auto_parallel_context(
            "parallel_mode") == context.ParallelMode.DATA_PARALLEL
        if self.is_data_parallel:
            self.allreduce = P.AllReduce()
            self.device_num = int(os.getenv("RANK_SIZE", "1"))

    def construct(self, pred, target, mask):
        """construct of MSELoss"""
        pred = self.cast(pred, mstype.float32)
        target = self.cast(target, mstype.float32)
        mask = self.cast(mask, mstype.float32)
        if self.norm_pixel_loss:
            mean = self.mean1(target, -1)
            var = self.variance(target)
            # var = target.var(keepdims=True, axis=-1)
            var = self.add_loss(var, 1e-6)
            std = self.pow(var, 0.5)
            sub = self.sub(target, mean)
            target = self.divide(sub, std)
        res = self.sub(pred, target)
        recon_loss = self.square(res)
        recon_loss = self.mean2(recon_loss, -1)
        loss_mask = self.mul(recon_loss, mask)
        loss_sum = self.sum(loss_mask)
        mask_sum = self.sum(mask)
        loss = self.divide1(loss_sum, mask_sum)
        if self.is_data_parallel:
            loss = self.allreduce(loss) / self.device_num
        return loss

    def variance(self, x):
        axis = (x.ndim - 1,)
        x_mean = self.mean1(x, axis)
        x_sub = self.sub(x, x_mean)
        x_pow = self.pow(x_sub, 2)
        x_sum = self.sum2(x_pow, axis)
        x_var = self.divide2(x_sum, x.shape[-1])
        return x_var


class SoftTargetCrossEntropy(LossBase):
    """SoftTargetCrossEntropy for MixUp Augment"""

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()
        self.mean_ops = P.ReduceMean(keep_dims=False)
        self.sum_ops = P.ReduceSum(keep_dims=False)
        self.mul = P.Mul()
        self.mul1d = P.Mul()
        self.log_softmax = P.LogSoftmax()

        self.is_data_parallel = context.get_auto_parallel_context(
            "parallel_mode") == context.ParallelMode.DATA_PARALLEL
        if self.is_data_parallel:
            self.allreduce = P.AllReduce()
            self.device_num = int(os.getenv("RANK_SIZE", "1"))

    def construct(self, logit, label):
        """construct of soft CE"""
        logit = P.Cast()(logit, mstype.float32)
        label = P.Cast()(label, mstype.float32)
        logit_softmax = self.log_softmax(logit)
        neg_target = self.mul1d(label, -1)
        soft_target = self.mul(neg_target, logit_softmax)
        loss = self.sum_ops(soft_target, -1)
        loss = self.mean_ops(loss)
        if self.is_data_parallel:
            loss = self.allreduce(loss) / self.device_num
        return loss


def get_loss(args):
    """get_loss"""
    loss = None
    if args.loss_type == 'soft_ce':
        loss = SoftTargetCrossEntropy()
    else:
        raise NotImplementedError

    return loss
