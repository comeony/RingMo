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
"""finetune of ringmo"""
import os
import argparse
import aicc_tools as ac

from mindspore import nn
from mindspore.train.model import Model

from mscv.lr import build_lr
from mscv.loss import build_loss
from mscv.optim import build_optim
from mscv.trainer import build_wrapper
from mscv.datasets import build_dataset
from mscv.tools.load_ckpt import load_ckpt
from mscv.tools.helper import str2bool, build_context
from mscv.models import build_model, build_eval_engine
from mscv.parallel_config import build_parallel_config
from mscv.monitors.callback import build_finetune_callback
from register.config import RingMoConfig, ActionDict


@ac.aicc_monitor
def main(args):
    # init context
    cfts, profile_cb = build_context(args)

    # train dataset
    args.logger.info(".........Build Training Dataset..........")
    train_dataset = build_dataset(args, is_pretrain=False)
    data_size = train_dataset.get_dataset_size()
    new_epochs = args.train_config.epoch
    if args.train_config.per_epoch_size and args.train_config.sink_mode:
        new_epochs = int((data_size / args.train_config.per_epoch_size) * new_epochs)
    else:
        args.train_config.per_epoch_size = data_size
    args.data_size = data_size
    args.logger.info("Will be Training epochs:{}， sink_size:{}".format(
        new_epochs, args.train_config.per_epoch_size))
    args.logger.info("Create training dataset finish, data size:{}".format(data_size))

    # evaluation dataset
    args.logger.info(".........Build Eval Dataset..........")
    eval_dataset = build_dataset(args, is_pretrain=False, is_train=False)

    # build context config
    args.logger.info(".........Build context config..........")
    build_parallel_config(args)
    args.logger.info("context config is:{}".format(args.parallel_config))
    args.logger.info("moe config is:{}".format(args.moe_config))

    # build net
    args.logger.info(".........Build Net..........")
    net = build_model(args)
    eval_engine = build_eval_engine(net, eval_dataset, args)

    # build lr
    args.logger.info(".........Build LR Schedule..........")
    lr_schedule = build_lr(args)
    args.logger.info("LR Schedule is: {}".format(args.lr_schedule))

    # define optimizer
    # layer-wise lr decay
    args.logger.info(".........Build Optimizer..........")
    optimizer = build_optim(args, net, lr_schedule, args.logger, is_pretrain=False)

    # define loss
    finetune_loss = build_loss(args)
    # Build train network
    net_with_loss = nn.WithLossCell(net, finetune_loss)
    net_with_train = build_wrapper(args, net_with_loss, optimizer, log=args.logger)

    # define Model and begin training
    args.logger.info(".........Starting Init Train Model..........")
    model = Model(net_with_train, metrics=eval_engine.metric, eval_network=eval_engine.eval_network)  #

    args.logger.info(".........Starting Init Eval Model..........")
    eval_engine.set_model(model)
    # equal to model._init(dataset, sink_size=per_step_size)
    eval_engine.compile(sink_size=args.train_config.per_epoch_size)

    # load pretrain or resume ckpt
    load_ckpt(args, cfts, net, model, net_with_train, train_dataset, new_epochs,
              is_finetune=True, valid_dataset=eval_dataset)

    # define callback
    callback = build_finetune_callback(args, cfts, eval_engine)

    if args.profile:
        callback.append(profile_cb)

    args.logger.info(".........Starting Training Model..........")
    model.train(new_epochs, train_dataset, callbacks=callback,
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
    parser.add_argument('--finetune_path', default=None, type=str, help='checkpoint path for finetune')
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
    if args_.finetune_path is not None:
        config.train_config.resume_ckpt = args_.finetune_path
    if args_.options is not None:
        config.merge_from_dict(args_.options)

    if config.finetune_dataset.eval_offset < 0:
        config.finetune_dataset.eval_offset = config.train_config.epoch % config.finetune_dataset.eval_interval

    main(config)
