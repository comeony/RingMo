from mindspore import nn
import mindspore.common.dtype as mstype
from mindspore.nn.transformer.moe import default_moe_config
from mindspore.nn.transformer.op_parallel_config import default_moeparallel_config

from .mlp import MLP


class Moe(nn.transformer.moe.MoE):
    def __init__(self,
                 hidden_size,
                 ffn_hidden_size,
                 dropout_rate,
                 use_dropout=True,
                 hidden_act='gelu',
                 weight_init='XavierUniform',
                 param_init_type=mstype.float32,
                 moe_config=default_moe_config,
                 parallel_config=default_moeparallel_config):
        super(Moe, self).__init__(
            hidden_size,
            ffn_hidden_size,
            dropout_rate,
            hidden_act=hidden_act,
            param_init_type=param_init_type,
            moe_config=moe_config,
            parallel_config=parallel_config)
        del self.ffn
        self.ffn = MLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            dropout_rate=dropout_rate,
            hidden_act=hidden_act,
            expert_num=self.expert_dim,
            weight_init=weight_init,
            use_dropout=use_dropout,
            param_init_type=param_init_type,
            parallel_config=parallel_config
        )
