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
"""pretrain of ringmo"""
import os
import argparse

from mindspore.train.model import Model
import aicc_tools as ac

from src.lr import build_lr
from src.arch import build_model
from src.optim import build_optim
from src.datasets import build_dataset
from src.trainer import build_wrapper
from src.parallel_config import build_parallel_config
from src.tools.helper import count_params
from src.monitors.callback import build_pretrain_callback
from src.tools.helper import build_context, str2bool
from src.tools.load_ckpt import load_ckpt
from register.config import RingMoConfig, ActionDict


@ac.aicc_monitor
def main(args):
    # init context
    cfts, profile_cb = build_context(args)

    # build dataset
    args.logger.info(".........Build Dataset..........")
    args.pretrain_dataset.data_path = cfts.get_dataset(args.pretrain_dataset.data_path)
    dataset = build_dataset(args)
    data_size = dataset.get_dataset_size()
    new_epochs = args.train_config.epoch
    if args.train_config.per_epoch_size and args.train_config.sink_mode:
        new_epochs = int((data_size / args.train_config.per_epoch_size) * new_epochs)
    else:
        args.train_config.per_epoch_size = data_size

    args.data_size = data_size
    args.logger.info("Will be Training epochs:{}， sink_size:{}".format(
        new_epochs, args.train_config.per_epoch_size))
    args.logger.info("Create training dataset finish, data size:{}".format(data_size))

    # build context config
    args.logger.info(".........Build context config..........")
    build_parallel_config(args)
    args.logger.info("context config is:{}".format(args.parallel_config))
    args.logger.info("moe config is:{}".format(args.moe_config))

    # build net
    args.logger.info(".........Build Net..........")
    net = build_model(args)
    args.logger.info("网络参数量：{} M.".format(count_params(net)))

    # build lr
    args.logger.info(".........Build LR Schedule..........")
    lr_schedule = build_lr(args)

    # define optimizer
    args.logger.info(".........Build Optimizer..........")
    optimizer = build_optim(args, net, lr_schedule, args.logger)

    # define model
    args.logger.info(".........Build Train Model..........")
    train_model = build_wrapper(args, net, optimizer, log=args.logger)
    args.logger.info("模型参数量：{} M.".format(count_params(train_model)))

    # define Model and begin training
    args.logger.info(".........Starting Init Train Model..........")
    model = Model(train_model)

    # resume ckpt
    load_ckpt(args, cfts, net, model, train_model, dataset, new_epochs)

    # define callback
    callback = build_pretrain_callback(args, cfts)

    if args.profile:
        callback.append(profile_cb)

    args.logger.info(".........Starting Training Model..........")
    model.train(new_epochs, dataset, callbacks=callback,
                dataset_sink_mode=args.train_config.sink_mode,
                sink_size=args.train_config.per_epoch_size)


if __name__ == "__main__":
    work_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default=os.path.join(work_path, "config path"),
        help='YAML config files')
    parser.add_argument('--device_id', default=None, type=int, help='device id')
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    parser.add_argument('--use_parallel', default=None, type=str2bool, help='whether use parallel mode')
    parser.add_argument('--profile', default=None, type=str2bool, help='whether use profile analysis')
    parser.add_argument(
        '--options',
        nargs='+',
        action=ActionDict,
        help='override some settings in the used config, the key-value pair'
             'in xxx=yyy format will be merged into config file')

    args_ = parser.parse_args()
    config = RingMoConfig(args_.config)
    if args_.device_id is not None:
        config.context.device_id = args_.device_id
    if args_.seed is not None:
        config.seed = args_.seed
    if args_.use_parallel is not None:
        config.use_parallel = args_.use_parallel
    if args_.profile is not None:
        config.profile = args_.profile
    if args_.options is not None:
        config.merge_from_dict(args_.options)

    main(config)
