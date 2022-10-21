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
"""metric"""
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.communication.management import GlobalComm


class ClassifyCorrect(nn.Cell):
    """ClassifyCorrectCell"""

    def __init__(self, network, use_moe=False):
        super(ClassifyCorrect, self).__init__(auto_prefix=False)
        self._network = network
        self.argmax = P.Argmax().shard(((1, 1),))
        self.equal = P.Equal().shard(((1,), (1,)))
        self.cast = P.Cast()
        self.reduce_sum = P.ReduceSum().shard(((1,),))
        self.use_moe = use_moe

    def construct(self, data, label):
        outputs = self._network(data)
        y_pred = self.argmax(outputs)
        y_pred = self.cast(y_pred, mstype.int32)
        y_correct = self.equal(y_pred, label)
        y_correct = self.cast(y_correct, mstype.float32)
        y_correct = self.reduce_sum(y_correct)
        return (y_correct,)


class ClassifyCorrectForDPMode(nn.Cell):
    """ClassifyCorrectCell"""

    def __init__(self, network, use_moe=False):
        super(ClassifyCorrectForDPMode, self).__init__(auto_prefix=False)
        self.use_moe = use_moe
        self._network = network
        self.argmax = P.Argmax().shard(((1, 1),))
        self.equal = P.Equal().shard(((1,), (1,)))
        self.cast = P.Cast()
        self.reduce_sum = P.ReduceSum().shard(((1,),))
        self.allreduce = P.AllReduce(P.ReduceOp.SUM, GlobalComm.WORLD_COMM_GROUP).shard(((),))

    def construct(self, data, label):
        """construct of ClassifyCorrectForDPMode"""
        if self.use_moe:
            outputs, _ = self._network(data)
        else:
            outputs = self._network(data)
        y_pred = self.argmax(outputs)
        y_pred = self.cast(y_pred, mstype.int32)
        y_correct = self.equal(y_pred, label)
        y_correct = self.cast(y_correct, mstype.float32)
        y_correct = self.reduce_sum(y_correct)
        total_correct = self.allreduce(y_correct)
        return (total_correct,)


class DistAccuracy(nn.Metric):
    """DistAccuracy"""

    def __init__(self, batch_size, device_num, samples_num: int = 50000):
        super(DistAccuracy, self).__init__()
        self.batch_size = batch_size
        self.device_num = device_num
        self.samples_num = samples_num
        self._correct_num = 0
        self._total_num = 0
        self.clear()

    def clear(self):
        self._correct_num = 0
        self._total_num = 0

    def update(self, *inputs):
        if len(inputs) != 1:
            raise ValueError('Distribute accuracy needs 1 input (y_correct), but got {}'.format(len(inputs)))
        y_correct = self._convert_data(inputs[0])
        self._correct_num += y_correct
        self._total_num += self.batch_size * self.device_num

    def eval(self):
        if self._total_num == 0:
            raise RuntimeError('Accuracy can not be calculated, because the number of samples is 0.')
        return self._correct_num / self.samples_num
