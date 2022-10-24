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
"""build optim"""
import json
from functools import partial

from mindspore import nn

from ringmo_framework.lr.lr_schedule import LearningRateWiseLayer
from .optimizer import AdamWeightDecayOp
from .optimizer import FP32StateAdamWeightDecay


def build_optim(config, model, lr, logger, is_pretrain=True):
    if is_pretrain:
        return build_pretrain_optimizer(config, model, lr, logger)
    return build_finetune_optimizer(config, model, lr, logger)


def build_pretrain_optimizer(config, model, lr, logger):
    """build pretrain optimizer"""
    logger.info('>>>>>>>>>> Build Optimizer for Pre-training Stage')
    optimizer_config = config.optimizer
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
        logger.info(f'No weight decay: {skip}')
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
        logger.info(f'No weight decay keywords: {skip_keywords}')

    group_parameters = get_pretrain_param_groups(optimizer_config, model, logger, skip, skip_keywords)

    optimizer = None
    if optimizer_config.optim_name == 'sgd':
        optimizer = nn.SGD(group_parameters, learning_rate=lr,
                           momentum=optimizer_config.momentum, weight_decay=optimizer_config.weight_decay)
    elif optimizer_config.optim_name == "AdamW":
        optimizer = nn.AdamWeightDecay(group_parameters,
                                       learning_rate=lr,
                                       weight_decay=optimizer_config.weight_decay,
                                       eps=optimizer_config.eps,
                                       beta1=optimizer_config.beta1,
                                       beta2=optimizer_config.beta2)
    elif optimizer_config.optim_name == "AdamWOP":
        optimizer = AdamWeightDecayOp(group_parameters,
                                      learning_rate=lr,
                                      weight_decay=optimizer_config.weight_decay,
                                      eps=optimizer_config.eps,
                                      beta1=optimizer_config.beta1,
                                      beta2=optimizer_config.beta2)
    elif optimizer_config.optim_name == "FP32AdamWOP":
        optimizer = FP32StateAdamWeightDecay(group_parameters,
                                             learning_rate=lr,
                                             weight_decay=optimizer_config.weight_decay,
                                             eps=optimizer_config.eps,
                                             beta1=optimizer_config.beta1,
                                             beta2=optimizer_config.beta2)
    else:
        raise NotImplementedError
    return optimizer


def build_finetune_optimizer(config, model, lr, logger):
    """build finetune optimizer"""
    logger.info('>>>>>>>>>> Build Optimizer for Fine-tuning Stage')
    optimizer_config = config.optimizer
    if config.model.backbone == 'swin':
        depths = config.model.depth
        num_layers = sum(depths)
        get_layer_func = partial(get_swin_layer, num_layers=num_layers + 2, depths=depths)
    elif config.model.backbone == 'vit':
        num_layers = config.model.depth
        get_layer_func = partial(get_vit_layer, num_layers=num_layers + 2)
    else:
        raise NotImplementedError

    scales = list(optimizer_config.layer_decay ** i for i in reversed(range(num_layers + 2)))

    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
        logger.info(f'No weight decay: {skip}')
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
        logger.info(f'No weight decay keywords: {skip_keywords}')

    group_parameters = get_finetune_param_groups(
        model, lr, optimizer_config.weight_decay,
        get_layer_func, scales, logger, skip, skip_keywords)

    optimizer = None
    if optimizer_config.optim_name == 'sgd':
        optimizer = nn.SGD(group_parameters,
                           learning_rate=lr,
                           momentum=optimizer_config.momentum,
                           weight_decay=optimizer_config.weight_decay)
    elif optimizer_config.optim_name == "AdamW":
        optimizer = nn.AdamWeightDecay(group_parameters,
                                       learning_rate=lr,
                                       weight_decay=optimizer_config.weight_decay,
                                       eps=optimizer_config.eps,
                                       beta1=optimizer_config.beta1,
                                       beta2=optimizer_config.beta2)
    elif optimizer_config.optim_name == "AdamWOP":
        optimizer = AdamWeightDecayOp(group_parameters,
                                      learning_rate=lr,
                                      weight_decay=optimizer_config.weight_decay,
                                      eps=optimizer_config.eps,
                                      beta1=optimizer_config.beta1,
                                      beta2=optimizer_config.beta2)
    else:
        raise NotImplementedError
    return optimizer


def get_pretrain_param_groups(config, model, logger, skip_list=(), skip_keywords=()):
    """get pretrain param groups"""
    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []

    for param in model.trainable_params():
        if len(param.shape) == 1 or param.name.endswith(".bias") or (param.name in skip_list) or \
                check_keywords_in_name(param.name, skip_keywords):
            no_decay.append(param)
            no_decay_name.append(param.name)
        else:
            has_decay.append(param)
            has_decay_name.append(param.name)
    logger.info(f'No decay params: {no_decay_name}')
    logger.info(f'Has decay params: {has_decay_name}')
    return [{'params': has_decay, 'weight_decay': config.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
            {'order_params': model.trainable_params()}]


def get_finetune_param_groups(model, base_lr, weight_decay, get_layer_func, scales, logger,
                              skip_list=(), skip_keywords=()):
    """get finetune param groups"""
    parameter_group_names = {}
    parameter_group_vars = {}

    for param in model.trainable_params():

        if len(param.shape) == 1 or param.name.endswith(".bias") or (param.name in skip_list) or \
                check_keywords_in_name(param.name, skip_keywords):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_layer_func is not None:
            layer_id = get_layer_func(param.name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales[layer_id]
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": LearningRateWiseLayer(base_lr, scale),
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(param.name)
    # parameter_group_vars["order_params"] = {"order_params": model.trainable_params()}
    logger.info("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def get_vit_layer(name, num_layers):
    """get vit layer"""
    if name in ("encoder.cls_tokens", "encoder.mask_token", "encoder.pos_embed"):
        return 0
    if name.startswith("encoder.patch_embed"):
        return 0
    if name.startswith("encoder.rel_pos_bias"):
        return num_layers - 1
    if name.startswith("encoder.encoder.blocks"):
        layer_id = int(name.split('.')[3])
        return layer_id + 1
    return num_layers - 1


def get_swin_layer(name, num_layers, depths):
    """get swin layer"""
    if name in ("encoder.mask_token",):
        return 0
    if name.startswith("encoder.patch_embed"):
        return 0
    if name.startswith("encoder.layers"):
        layer_id = int(name.split('.')[2])
        block_id = name.split('.')[4]
        if block_id in ('reduction', 'norm'):
            return sum(depths[:layer_id + 1])
        layer_id = sum(depths[:layer_id]) + int(block_id)
        return layer_id + 1
    return num_layers - 1


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
