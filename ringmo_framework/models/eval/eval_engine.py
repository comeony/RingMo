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
"""eval engine"""

from mindspore import context

from ringmo_framework.models.eval.metric import ClassifyCorrect, ClassifyCorrectForDPMode, DistAccuracy


class BasicEvalEngine():
    """BasicEvalEngine"""

    def __init__(self):
        self.model = None

    @property
    def metric(self):
        return None

    @property
    def eval_network(self):
        return None

    def compile(self, sink_size=-1):
        pass

    def eval(self):
        pass

    def set_model(self, model):
        self.model = model

    def get_result(self):
        return None


class EvelEngine(BasicEvalEngine):
    """ImageNetEvelEngine"""

    def __init__(self, net, eval_dataset, args):
        super(EvelEngine, self).__init__()
        self.eval_dataset = eval_dataset
        is_data_parallel = context.get_auto_parallel_context(
            "parallel_mode") == context.ParallelMode.DATA_PARALLEL
        use_moe = args.moe_config.expert_num > 1
        if is_data_parallel:
            self.dist_eval_network = ClassifyCorrectForDPMode(net, use_moe)
        else:
            self.dist_eval_network = ClassifyCorrect(net, use_moe)
        self.args = args
        self.outputs = None
        self.model = None

    @property
    def metric(self):
        return {'acc': DistAccuracy(batch_size=self.args.train_config.batch_size,
                                    device_num=self.args.device_num,
                                    samples_num=self.args.finetune_dataset.samples_num)}

    @property
    def eval_network(self):
        return self.dist_eval_network

    def eval(self):
        self.outputs = self.model.eval(self.eval_dataset)

    def get_result(self):
        return self.outputs["acc"]


def build_eval_engine(net, eval_dataset, args):
    """get_eval_engine"""
    eval_engine = EvelEngine(net, eval_dataset, args)
    return eval_engine
