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
"""vit"""
import math
import numpy as np

from mindspore import nn
from mindspore import Tensor
from mindspore import ops as P
from mindspore import dtype as mstype
from mindspore.context import ParallelMode
from mindspore.common.parameter import Parameter
import mindspore.common.initializer as weight_init
from mindspore.train.serialization import load_param_into_net
from mindspore.nn.transformer.transformer import TransformerOpParallelConfig, \
    TransformerRecomputeConfig, default_moe_config

from ringmo_framework.models.layers.patch import PatchEmbed
from ringmo_framework.models.layers.vision_transformer import VisionTransformer
from ringmo_framework.models.core.relative_pos_bias import RelativePositionBias
from ringmo_framework.models.layers.layers import LayerNorm, Linear, Dropout, Identity

PARALLELMODE = (ParallelMode.AUTO_PARALLEL, ParallelMode.SEMI_AUTO_PARALLEL,)
default_recompute_config = TransformerRecomputeConfig()
default_parallel_config = TransformerOpParallelConfig(recompute=default_recompute_config)


class Vit(nn.Cell):
    """Vision Transformer with support for patch or hybrid CNN input stage."""

    def __init__(self,
                 parallel_config=default_parallel_config,
                 moe_config=default_moe_config,
                 batch_size=32, image_size=224, patch_size=16, in_chans=3,
                 num_classes=1000, embed_dim=768, depth=12, patch_type="conv",
                 num_heads=12, mlp_ratio=4, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., init_values=0., use_abs_pos_emb=True, use_rel_pos_bias=False,
                 use_shared_rel_pos_bias=False, use_mean_pooling=False, predictor_layer=False):
        super(Vit, self).__init__()
        self.use_moe = (moe_config.expert_num > 1)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.batch_size = batch_size
        self.image_size = image_size
        self.mlp_ratio = mlp_ratio
        self.parallel_config = parallel_config

        dp = parallel_config.data_parallel

        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size,
                                      in_features=in_chans, out_features=embed_dim,
                                      patch_type=patch_type,
                                      parallel_config=parallel_config)
        num_patches = self.patch_embed.num_patches

        self.cls_tokens = Parameter(
            weight_init.initializer(weight_init.Normal(sigma=.02), (1, 1, embed_dim)),
            name='cls', requires_grad=True)

        self.num_patches = num_patches
        seq_length = num_patches + 1
        if use_abs_pos_emb:
            self.pos_embed = Parameter(
                weight_init.initializer(weight_init.TruncatedNormal(sigma=.02), (1, seq_length, embed_dim)),
                name='pos_embedding', requires_grad=True)
        else:
            self.pos_embed = None

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.patch_embed.grid_size, num_heads=num_heads)
        else:
            self.rel_pos_bias = None
        self.use_rel_pos_bias = use_rel_pos_bias

        self.encoder_config = {
            'batch_size': self.batch_size, 'num_layers': depth,
            'num_heads': num_heads, 'hidden_size': embed_dim,
            'ffn_hidden_size': int(embed_dim * mlp_ratio),
            'predictor_layer': predictor_layer,
            'seq_length': seq_length,
            'weight_init': weight_init.TruncatedNormal(sigma=.02),
            'window_size': self.patch_embed.grid_size if use_rel_pos_bias else None,
            'drop_rate': drop_rate,
            'hidden_dropout_rate': drop_path_rate,
            'init_values': init_values,
            'attention_dropout_rate': attn_drop_rate,
            'moe_config': moe_config,
            'parallel_config': parallel_config
        }
        self.seq_length = seq_length
        self.encoder = VisionTransformer(**self.encoder_config)
        self.encoder_input_mask = Tensor(np.ones((batch_size, seq_length, seq_length)),
                                         mstype.float32)

        self.add = P.Add().shard(((dp, 1, 1), (1, 1, 1)))
        self.cast = P.Cast()
        self.tile = P.Tile().shard(((dp, 1, 1),))
        self.cat = P.Concat(axis=1)
        self.norm = LayerNorm((embed_dim,), eps=1e-6).shard(((dp, 1, 1),))
        if use_mean_pooling:
            self.fc_norm = LayerNorm((embed_dim,), eps=1e-6).shard(((dp, 1),))
            del self.norm
        else:
            self.fc_norm = Identity()

        self.global_pool = use_mean_pooling
        self.reduce_mean = P.ReduceMean().shard(((dp, 1, 1),))
        self.dropout = Dropout(keep_prob=(1. - drop_rate))
        self.dropout.shard(((dp, 1, 1),))

        self.stride_slice = P.StridedSlice().shard(((dp, 1, 1),))

        self.init_weights_vit()
        self.fix_init_weight()

    def fix_init_weight(self):
        """fix init weight"""
        def rescale(param, layer_id):
            values = param.data / (math.sqrt(2.0 * layer_id))
            param.set_data(values)

        for layer_id, block in enumerate(self.encoder.blocks):  # check if use_moe
            if self.use_moe:
                rescale(block.attention.projection.weight, layer_id + 1)
                rescale(block.output.ffn.projection.weight, layer_id + 1)
            else:
                rescale(block.attention.projection.weight, layer_id + 1)
                rescale(block.output.projection.weight, layer_id + 1)

    def init_weights_vit(self):
        """init weights vit
         ViT weight initialization, original timm impl (for reproducibility) """
        for _, cell in self.cells_and_names():

            if isinstance(cell, Linear):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, Linear) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, (LayerNorm, nn.LayerNorm)):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))

    def no_weight_decay(self):
        return {'pos_embed', 'cls_tokens'}

    def load_pretrained(self, params_dict):
        return load_param_into_net(self, params_dict)

    def construct(self, img):
        """construct of vit"""
        tokens = self.patch_embed(img)
        cls_tokens = self.tile(self.cls_tokens, (self.batch_size, 1, 1))
        tokens = self.cat((cls_tokens, tokens))
        if self.pos_embed is not None:
            tokens = self.add(tokens, self.pos_embed)

        tokens = self.dropout(tokens)

        if self.rel_pos_bias:
            rel_pos_bias = self.rel_pos_bias()
            x = self.encoder(tokens, self.encoder_input_mask, rel_pos_bias=rel_pos_bias)
        else:
            x = self.encoder(tokens, self.encoder_input_mask)

        b, s, c = x.shape

        if self.global_pool:
            x = self.stride_slice(
                x, (0, 1, 0), (b, s, c), (1, 1, 1)
            )
            x = self.reduce_mean(x, 1)
            out = self.fc_norm(x)
        else:
            out = self.stride_slice(
                x, (0, 0, 0), (b, 1, c), (1, 1, 1)
            )
        return out


class FinetuneVit(nn.Cell):
    """finetune vit"""
    def __init__(self, **kwargs):
        super(FinetuneVit, self).__init__()
        self.encoder = Vit(**kwargs)
        self.use_moe = self.encoder.encoder_config["moe_config"].expert_num > 1
        self.head = Linear(
            self.encoder.embed_dim, self.encoder.num_classes,
            weight_init=weight_init.TruncatedNormal(sigma=2e-5),
            compute_dtype=mstype.float32).to_float(mstype.float32)

    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    def load_pretrained(self, params_dict):
        return load_param_into_net(self, params_dict)

    def construct(self, img):
        x = self.encoder(img)
        return self.head(x)


def vit_base_p16(**kwargs):
    return Vit(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs)


def vit_large_p16(**kwargs):
    return Vit(patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, **kwargs)


def vit_huge_p14(**kwargs):
    return Vit(patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, **kwargs)


def build_vit(config):
    """build vit"""
    model = FinetuneVit(
        parallel_config=config.parallel_config,
        moe_config=config.moe_config,
        batch_size=config.train_config.batch_size * config.device_num
        if config.parallel.parallel_mode == "semi_auto_parallel" else config.train_config.batch_size,
        image_size=config.train_config.image_size,
        patch_size=config.model.patch_size,
        in_chans=config.model.in_chans,
        num_classes=config.train_config.num_classes,
        embed_dim=config.model.embed_dim,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        drop_rate=config.model.drop_rate,
        drop_path_rate=config.model.drop_path_rate,
        use_abs_pos_emb=config.model.use_abs_pos_emb,
        init_values=config.model.init_values,
        use_rel_pos_bias=config.model.use_rel_pos_bias,
        use_shared_rel_pos_bias=config.model.use_shared_rel_pos_bias,
        use_mean_pooling=config.model.use_mean_pooling)
    return model
