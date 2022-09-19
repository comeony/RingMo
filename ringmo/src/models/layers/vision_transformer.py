import numpy as np

from mindspore import nn
import mindspore.common.dtype as mstype
from mindspore.nn.transformer.moe import default_moe_config
from mindspore.nn.transformer.transformer import default_transformer_config, _get_lambda_func

from .block import Block


class VisionTransformer(nn.transformer.transformer.TransformerEncoder):
    r"""
        VisionTransformer module with multi-layer stacked of `TransformerLayer`, including multihead self
        attention and feedforward layer.
    """

    def __init__(self,
                 batch_size,
                 num_layers,
                 hidden_size,
                 ffn_hidden_size,
                 seq_length,
                 num_heads,
                 predictor_layer=False,
                 window_size=None,
                 drop_rate=0.,
                 attention_dropout_rate=0.,
                 hidden_dropout_rate=0.,
                 hidden_act='gelu',
                 weight_init='XavierUniform',
                 init_values=None,
                 post_layernorm_residual=False,
                 layernorm_compute_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 lambda_func=None,
                 offset=0,
                 moe_config=default_moe_config,
                 parallel_config=default_transformer_config):
        super(VisionTransformer, self).__init__(
            batch_size,
            num_layers,
            hidden_size,
            ffn_hidden_size,
            seq_length,
            num_heads,
            attention_dropout_rate=attention_dropout_rate,
            hidden_dropout_rate=hidden_dropout_rate,
            hidden_act=hidden_act,
            post_layernorm_residual=post_layernorm_residual,
            layernorm_compute_type=layernorm_compute_type,
            softmax_compute_type=softmax_compute_type,
            param_init_type=param_init_type,
            lambda_func=lambda_func,
            offset=offset,
            moe_config=moe_config,
            parallel_config=parallel_config
        )
        hdr = [x.item() for x in np.linspace(0, hidden_dropout_rate, num_layers)]  # stochastic depth decay rule
        self.batch_size = batch_size
        self.blocks = nn.CellList()
        parallel_config_args = parallel_config.moe_parallel_config if self.use_moe else parallel_config.dp_mp_config
        for i in range(num_layers):
            block = Block(
                hidden_size=hidden_size,
                batch_size=batch_size,
                ffn_hidden_size=ffn_hidden_size,
                seq_length=seq_length,
                drop_rate=drop_rate,
                attention_dropout_rate=attention_dropout_rate,
                hidden_dropout_rate=hdr[i],
                init_values=init_values,
                weight_init=weight_init,
                layernorm_compute_type=layernorm_compute_type,
                softmax_compute_type=softmax_compute_type,
                window_size=window_size,
                num_heads=num_heads,
                hidden_act=hidden_act,
                post_layernorm_residual=post_layernorm_residual,
                param_init_type=param_init_type,
                moe_config=moe_config,
                parallel_config=parallel_config_args)
            # If the user doesn't pass the fusion function, use the default one
            if not lambda_func:
                lambda_func = _get_lambda_func()

            lambda_func(block, layer_id=i, layers=num_layers,
                        offset=offset, parallel_config=parallel_config)
            self.blocks.append(block)

    def construct(self, hidden_states, attention_mask, init_reset=True, batch_valid_length=None, rel_pos_bias=None):
        output = ()
        if self.use_moe:
            accum_loss = self.aux_loss
            for i in range(self.num_layers):
                hidden_states, aux_loss = self.blocks[i](
                    hidden_states, attention_mask, init_reset, batch_valid_length, rel_pos_bias)

                accum_loss = self.add(accum_loss, aux_loss)
            output = output + (hidden_states, accum_loss,)
            return output

        for i in range(self.num_layers):
            hidden_states = self.blocks[i](
                hidden_states, attention_mask, init_reset, batch_valid_length, rel_pos_bias)
        output = hidden_states
        return output
