# Copyright 2021 Huawei Technologies Co., Ltd
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
from mindspore.common import dtype as mstype
from mindspore.nn.loss.loss import LossBase
from mindspore import ops as P


class SoftTargetCrossEntropy(LossBase):
    """SoftTargetCrossEntropy for MixUp Augment"""

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()
        self.mean_ops = P.ReduceMean(keep_dims=False)
        self.sum_ops = P.ReduceSum(keep_dims=False)
        self.mul = P.Mul()
        self.mul1d = P.Mul()
        self.log_softmax = P.LogSoftmax()

    def construct(self, logit, label):
        logit = P.Cast()(logit, mstype.float32)
        label = P.Cast()(label, mstype.float32)
        logit_softmax = self.log_softmax(logit)
        neg_target = self.mul1d(label, -1)
        soft_target = self.mul(neg_target, logit_softmax)
        loss = self.sum_ops(soft_target, -1)
        return self.mean_ops(loss)


def get_loss(args):
    """get_loss"""
    loss = None
    if args.loss_type == 'soft_ce':
        loss = SoftTargetCrossEntropy()
    else:
        raise NotImplementedError

    return loss
