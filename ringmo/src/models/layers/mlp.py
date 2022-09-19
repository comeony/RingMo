from mindspore import nn
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F
from mindspore.nn.transformer.op_parallel_config import default_dpmp_config

from .layers import Linear, Dropout


class MLP(nn.transformer.transformer.FeedForward):
    r"""MLP for ring-mo."""

    def __init__(self,
                 hidden_size,
                 ffn_hidden_size=None,
                 out_features=None,
                 dropout_rate=0.,
                 hidden_act='gelu',
                 expert_num=1,
                 weight_init='XavierUniform',
                 use_dropout=True,
                 param_init_type=mstype.float32,
                 parallel_config=default_dpmp_config):
        ffn_hidden_size = ffn_hidden_size or hidden_size
        super(MLP, self).__init__(
            hidden_size,
            ffn_hidden_size,
            dropout_rate=dropout_rate,
            hidden_act=hidden_act,
            expert_num=expert_num,
            param_init_type=mstype.float32,
            parallel_config=parallel_config)
        mp = parallel_config.model_parallel
        if expert_num > 1:
            ep = parallel_config.expert_parallel
        else:
            ep = 1
        # ffn use less dp than other ops when use_moe, due to there are ops use dp and ep.
        dp = int(parallel_config.data_parallel / ep)
        out_features = out_features or hidden_size

        # Project to ffn_hidden_size
        self.mapping = Linear(in_channels=hidden_size,
                              out_channels=ffn_hidden_size,
                              activation=hidden_act,
                              transpose_b=False,
                              expert_num=expert_num,
                              outer_batch=dp,
                              weight_init=weight_init,
                              param_init_type=param_init_type)

        if expert_num > 1:
            self.mapping.shard(strategy_matmul=((dp, ep, 1, 1), (ep, 1, mp)),
                               strategy_bias=((dp, ep, 1, mp), (mp,)),
                               strategy_activation=((dp, ep, 1, mp),))
        else:
            self.mapping.shard(strategy_matmul=((dp, 1), (1, mp)),
                               strategy_bias=((dp, mp), (mp,)),
                               strategy_activation=((dp, mp),))
        # Project back to hidden_size
        self.projection = Linear(in_channels=ffn_hidden_size,
                                 out_channels=out_features,
                                 transpose_b=False,
                                 expert_num=expert_num,
                                 outer_batch=dp,
                                 weight_init=weight_init,
                                 param_init_type=param_init_type)
        if expert_num > 1:
            self.projection.shard(strategy_matmul=((dp, ep, 1, mp), (ep, mp, 1)),
                                  strategy_bias=((dp, ep, 1, 1), (1,)))
        else:
            self.projection.shard(strategy_matmul=((dp, mp), (mp, 1)),
                                  strategy_bias=((dp, 1), (1,)))
        self.projection.bias.parallel_optimizer = False
        self.dropout = Dropout(1 - dropout_rate)
        self.dropout.shard(((dp, 1),))
        self.dropout_3d = Dropout(1 - dropout_rate)
        self.dropout_3d.shard(((dp, 1, 1),))
        self.dropout_4d = Dropout(1 - dropout_rate)
        self.dropout_4d.shard(((dp, ep, 1, 1),))
        self.use_dropout = use_dropout

    def construct(self, x):
        x = self.cast(x, mstype.float16)
        # returned shape is [bs, seq_length, ffn_hidden_size] or [bs * seq_length, ffn_hidden_size]
        hidden = self.mapping(x)

        if self.use_dropout:
            if len(F.shape(hidden)) == 3:
                hidden = self.dropout_3d(hidden)
            elif len(F.shape(hidden)) == 2:
                hidden = self.dropout(hidden)
            else:
                hidden = self.dropout_4d(hidden)

        output = self.projection(hidden)
        # returned shape is [bs, seq_length, ffn_hidden_size] or [bs * seq_length, ffn_hidden_size]
        if len(F.shape(output)) == 3:
            output = self.dropout_3d(output)
        elif len(F.shape(output)) == 2:
            output = self.dropout(output)
        else:
            output = self.dropout_4d(output)
        return output
