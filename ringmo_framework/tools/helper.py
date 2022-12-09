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
"""helper of ringmo"""
import os
import sys
import yaml
import numpy as np

from mindspore import context
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs

import aicc_tools as ac


def build_context(args):
    """build context"""
    profile_cb = None
    if args.profile and args.use_parallel:
        cfts_1 = ac.CFTS(**args.aicc_config)
        profile_cb = cfts_1.profile_monitor(start_step=args.profile_start_step, stop_step=args.profile_end_step)
        args.train_config.sink_mode = False

    args.seed = int(os.getenv("RANK_ID", "0")) + args.seed
    local_rank, device_num = ac.context_init(seed=args.seed, use_parallel=args.use_parallel,
                                             context_config=args.context, parallel_config=args.parallel)
    context.set_context(max_device_memory="30GB")
    # context.set_context(env_config_path=args.mem_reuse)
    set_algo_parameters(elementwise_op_strategy_follow=True, fully_use_devices=False)
    _set_multi_subgraphs()

    args.device_num = device_num
    args.local_rank = local_rank
    args.logger = ac.get_logger()
    args.logger.info("model config: {}".format(args))

    # init cfts
    cfts = ac.CFTS(**args.aicc_config, rank_id=local_rank)

    if args.parallel.get("strategy_ckpt_load_file"):
        args.parallel["strategy_ckpt_load_file"] = cfts.get_checkpoint(args.parallel.get("strategy_ckpt_load_file"))
        context.set_auto_parallel_context(strategy_ckpt_load_file=args.parallel["strategy_ckpt_load_file"])

    if args.profile and not args.use_parallel:
        cfts_2 = ac.CFTS(**args.aicc_config)
        profile_cb = cfts_2.profile_monitor(start_step=args.profile_start_step, stop_step=args.profile_end_step)
        args.train_config.sink_mode = False
    return cfts, profile_cb


def str2bool(b):
    if b.lower() in ["false"]:
        output = False
    elif b.lower() in ["true"]:
        output = True
    else:
        raise Exception("Invalid Bool Value")
    return output


def parse_with_config(parser):
    """Parse With Config"""
    args = parser.parse_args()
    if args.config is not None:
        config_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
        override_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:]
                         if arg.startswith('--')}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config
    return args


def count_params(net):
    """Count number of parameters in the network
    Args:
        net (mindspore.nn.Cell): Mindspore network instance
    Returns:
        total_params (int): Total number of trainable params
    """
    total_params = [np.prod(param.shape) for param in net.trainable_params()]
    return sum(total_params) // 1000000
