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
"""build lr"""
from ringmo_framework.lr.lr_schedule import CosineDecayLR, WarmUpLR, \
    WarmUpCosineDecayV2, WarmUpCosineDecayV1, WarmUpMultiStepDecay


def build_lr(config):
    """build lr"""
    lr_config = config.lr_schedule
    device_num = config.device_num
    batch_size = config.train_config.batch_size
    _check_lr_config(lr_config, device_num=device_num, batch_size=batch_size, arch=config.arch)
    total_epochs = config.train_config.epoch
    warmup_epochs = lr_config.warmup_epochs
    steps_per_epoch = config.data_size
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    decay_steps = total_steps - warmup_steps

    lr_type = lr_config.lr_type
    lr = None
    if lr_type == 'cosine_decay':
        lr = CosineDecayLR(
            lr_config.min_lr, lr_config.base_lr, decay_steps)
    if lr_type == 'warmup':
        lr = WarmUpLR(
            lr_config.base_lr, warmup_steps, lr_config.warmup_lr)
    if lr_type == 'warmup_cosine_decay':
        lr = WarmUpCosineDecayV1(
            lr_config.min_lr, lr_config.base_lr, warmup_steps, decay_steps, lr_config.warmup_lr)
    if lr_type == 'warmup_cosine_decay_simmim':
        lr = WarmUpCosineDecayV2(
            lr_config.base_lr, total_steps, lr_config.min_lr,
            warmup_t=warmup_steps, warmup_lr_init=lr_config.warmup_lr)
    if lr_type == 'warmup_multistep_decay':
        lr = WarmUpMultiStepDecay(lr_config.base_lr, warmup_steps, lr_config.warmup_lr,
                                  lr_config.factor, lr_config.multi_epochs, steps_per_epoch)
    return lr


def _check_lr_config(config, device_num=1, batch_size=128, arch="simmim"):
    if arch in ('simmim', "ringmo", "ringmo_mm"):
        config.base_lr = (config.base_lr * device_num * batch_size) / 512
        config.min_lr = (config.min_lr * device_num * batch_size) / 512
        config.warmup_lr = (config.warmup_lr * device_num * batch_size) / 512
    if arch == 'mae':
        # base_lr(5e-4) * device_num * batch_size / 256
        config.base_lr = (config.base_lr * device_num * batch_size) / 256
