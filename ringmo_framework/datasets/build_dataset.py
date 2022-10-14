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
"""build dataset"""
from ringmo_framework.datasets.pretrain_dataset import create_pretrain_dataset
from ringmo_framework.datasets.finetune_dataset import create_finetune_dataset


def build_dataset(config, is_pretrain=True, is_train=True):
    if is_pretrain:
        return create_pretrain_dataset(config)
    return create_finetune_dataset(config, is_train=is_train)
