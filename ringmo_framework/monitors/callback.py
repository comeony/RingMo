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
"""callback of ringmo"""
from .monitor import StateMonitor


def build_pretrain_callback(args, cfts):
    """build pretrain callback"""
    train_config = args.train_config
    ckpt_config = args.callback.ckpt_config
    summary_config = args.callback.summary_config

    loss_cb = cfts.loss_monitor(per_print_times=1)
    summary_cb = cfts.summary_monitor(**summary_config)
    ckpt_append_info = [
        {"epoch_num": train_config.has_trained_epoches,
         "step_num": train_config.has_trained_steps}
    ]
    ckpt_cb = cfts.checkpoint_monitor(prefix=ckpt_config.prefix + "_rank_{}".format(args.local_rank),
                                      save_checkpoint_steps=ckpt_config.save_ckpt_epochs * train_config.per_epoch_size,
                                      keep_checkpoint_max=ckpt_config.keep_checkpoint_max,
                                      integrated_save=ckpt_config.integrated_save,
                                      async_save=ckpt_config.async_save,
                                      append_info=ckpt_append_info)
    obs_cb = cfts.obs_monitor()
    callback = [loss_cb, ckpt_cb, summary_cb, obs_cb]
    return callback


def build_finetune_callback(args, cfts, eval_engine):
    """build finetune callback"""
    ckpt_config = args.callback.ckpt_config
    train_config = args.train_config
    dataset_config = args.finetune_dataset
    state_cb = StateMonitor(data_size=train_config.per_epoch_size,
                            tot_batch_size=train_config.batch_size * args.device_num,
                            eval_interval=dataset_config.eval_interval,
                            eval_offset=dataset_config.eval_offset,
                            eval_engine=eval_engine,
                            logger=args.logger.info)

    ckpt_append_info = [{"epoch_num": train_config.has_trained_epoches,
                         "step_num": train_config.has_trained_steps}]
    ckpt_cb = cfts.checkpoint_monitor(prefix=ckpt_config.prefix + "_rank_{}".format(args.local_rank),
                                      save_checkpoint_steps=ckpt_config.save_ckpt_epochs * train_config.per_epoch_size,
                                      keep_checkpoint_max=ckpt_config.keep_checkpoint_max,
                                      integrated_save=ckpt_config.integrated_save,
                                      async_save=ckpt_config.async_save,
                                      append_info=ckpt_append_info)
    obs_cb = cfts.obs_monitor()
    callback = [ckpt_cb, state_cb, obs_cb]  #
    return callback
