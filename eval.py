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
"""eval of ringmo"""
import os
import argparse
import aicc_tools as ac

from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint

from ringmo_framework.datasets import build_dataset
from ringmo_framework.tools.helper import str2bool, build_context
from ringmo_framework.models import build_model, build_eval_engine
from ringmo_framework.parallel_config import build_parallel_config
from register.config import RingMoConfig


@ac.aicc_monitor
def main(args):
    # init context
    cfts, _ = build_context(args)

    # evaluation dataset
    args.logger.info(".........Build Eval Dataset..........")
    args.finetune_dataset.eval_path = cfts.get_dataset(args.finetune_dataset.eval_path)
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

    # load task ckpt
    resume_ckpt = args.train_config.resume_ckpt
    if resume_ckpt:
        args.logger.info(".........Load Task Checkpoint..........")
        resume_ckpt = cfts.get_checkpoint(resume_ckpt)
        params_dict = load_checkpoint(resume_ckpt, filter_prefix=["adam_m", "adam_v"])
        net_not_load = net.load_pretrained(params_dict)
        args.logger.info(f"===============net_not_load================{net_not_load}")

    args.logger.info(".........Starting Init Eval Model..........")
    model = Model(net, metrics=eval_engine.metric, eval_network=eval_engine.eval_network)
    eval_engine.set_model(model)
    # define Model and begin eval
    args.logger.info(".........Starting Eval Model..........")
    eval_engine.eval()
    output = eval_engine.get_result()
    last_metric = 'Top1 accuracy={:.6f}'.format(float(output))
    args.logger.info(last_metric)


if __name__ == "__main__":
    work_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default=os.path.join(work_path, "../config/simmim/aircas/vit/pretrain-simmim-vit-moe-p16-01.yaml"),
        help='YAML config files')
    parser.add_argument('--device_id', default=None, type=int, help='device id')
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    parser.add_argument('--batch_size', default=None, type=int, help='batch size')
    parser.add_argument('--use_parallel', default=None, type=str2bool, help='whether use parallel mode')
    parser.add_argument('--eval_path', default=None, type=str, help='checkpoint path for eval')

    args_ = parser.parse_args()
    config = RingMoConfig(args_.config)
    if args_.device_id is not None:
        config.context.device_id = args_.device_id
    if args_.seed is not None:
        config.seed = args_.seed
    if args_.use_parallel is not None:
        config.use_parallel = args_.use_parallel
    if args_.eval_path is not None:
        config.train_config.resume_ckpt = args_.eval_path
    if args_.batch_size is not None:
        config.train_config.batch_size = args_.batch_size

    if config.finetune_dataset.eval_offset < 0:
        config.finetune_dataset.eval_offset = config.train_config.epoch % config.finetune_dataset.eval_interval

    main(config)
