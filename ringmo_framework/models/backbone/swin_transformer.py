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
"""Define SwinTransformer model"""

import numpy as np

from mindspore import Parameter
from mindspore import nn, Tensor
import mindspore.ops.operations as P
from mindspore import dtype as mstype
import mindspore.common.initializer as weight_init_
from mindspore.train.serialization import load_param_into_net
from mindspore.nn.transformer.moe import default_moe_config
from mindspore.nn.transformer.op_parallel_config import default_dpmp_config

from ringmo_framework.models.layers.layers import LayerNorm, Linear, Dropout
from ringmo_framework.models.layers.patch import PatchEmbed
from ringmo_framework.models.layers.block import SwinTransformerBlock


class PatchMerging(nn.Cell):
    """ Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm/_LayerNorm
    """

    def __init__(self,
                 input_resolution,
                 dim,
                 weight_init='normal',
                 norm_layer=LayerNorm,
                 parallel_config=default_dpmp_config):
        super(PatchMerging, self).__init__()
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.input_resolution = input_resolution
        self.dim = dim[0] if isinstance(dim, tuple) and len(dim) == 1 else dim
        # Default False
        self.reduction = Linear(
            in_channels=4 * dim, out_channels=2 * dim, has_bias=False, weight_init=weight_init).to_float(mstype.float16)
        self.reduction.shard(strategy_matmul=((dp, mp), (mp, 1)), strategy_bias=((dp, 1), (1,)))
        self.norm = norm_layer([dim * 4,], eps=1e-4)
        self.norm.shard(((dp, 1, 1),))
        self.h, self.w = self.input_resolution
        self.h_2, self.w_2 = self.h // 2, self.w // 2
        self.h2w2 = int(self.h * self.w // 4)
        self.dim_mul_4 = int(dim * 4)
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(((dp, 1, 1, 1, 1, 1),))

    def construct(self, x):
        """
        x: B, H*W, C
        """
        b = x.shape[0]
        x = self.reshape(x, (b, self.h_2, 2, self.w_2, 2, self.dim))
        x = self.transpose(x, (0, 1, 3, 4, 2, 5))
        x = self.reshape(x, (b, self.h2w2, self.dim_mul_4))
        x = self.norm(x)
        x = self.cast(x, mstype.float16)
        x = self.reduction(x)
        x = self.cast(x, mstype.float32)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class SwinBasicLayer(nn.Cell):
    """ Swin Basic Layer
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=LayerNorm, downsample=None,
                 weight_init='normal', moe_config=default_moe_config,
                 parallel_config=default_dpmp_config):
        super(SwinBasicLayer, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.CellList([
            SwinTransformerBlock(dim=dim,
                                 input_resolution=input_resolution,
                                 num_heads=num_heads,
                                 window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 drop=drop,
                                 attn_drop=attn_drop,
                                 weight_init=weight_init,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 moe_config=moe_config,
                                 parallel_config=parallel_config)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer, parallel_config=parallel_config)
        else:
            self.downsample = None

    def construct(self, x):
        """construct"""
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class SwinTransformer(nn.Cell):
    """ Swin Transformer Model
    """

    def __init__(self, batch_size=None, image_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=None, num_heads=None, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=LayerNorm, ape=False, patch_norm=True, patch_type="conv",
                 moe_config=default_moe_config,
                 parallel_config=default_dpmp_config):
        super(SwinTransformer, self).__init__()
        dp = parallel_config.data_parallel
        self.parallel_config = parallel_config
        self.use_moe = moe_config.expert_num > 1
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.batch_size = batch_size
        self.cast = P.Cast()

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=image_size, patch_size=patch_size, in_features=in_chans, out_features=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None, patch_type=patch_type,
            parallel_config=parallel_config)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        # patches_resolution = self.patch_embed.patches_resolution
        patches_resolution = self.patch_embed.grid_size
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = Parameter(
                Tensor(np.zeros((1, num_patches, embed_dim)), dtype=mstype.float32), name="ape")

        self.pos_drop = Dropout(keep_prob=1.0 - drop_rate)
        self.pos_drop.shard(((dp, 1, 1),))

        # stochastic depth
        dpr = list(np.linspace(0, drop_path_rate, sum(depths)))  # stochastic depth decay rule
        parallel_config_args = parallel_config.moe_parallel_config if self.use_moe else parallel_config.dp_mp_config

        # build layers
        self.layers = nn.CellList()
        self.final_seq = num_patches  # downsample seq_length
        for i_layer in range(self.num_layers):
            layer = SwinBasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                moe_config=moe_config, parallel_config=parallel_config_args)
            # downsample seq_length
            if i_layer < self.num_layers - 1:
                self.final_seq = self.final_seq // 4
            self.layers.append(layer)

        self.norm = norm_layer([self.num_features,], eps=1e-6).shard(((dp, 1, 1),))
        self.transpose = P.Transpose().shard(((dp, 1, 1),))
        self.avgpool = P.ReduceMean(keep_dims=False).shard(((dp, 1, 1),))
        self.init_weights()

    def init_weights(self):
        """ ViT weight initialization, original timm impl (for reproducibility) """
        for _, cell in self.cells_and_names():
            if isinstance(cell, Linear):
                cell.weight.set_data(weight_init_.initializer(
                    weight_init_.TruncatedNormal(sigma=0.02),
                    cell.weight.shape,
                    cell.weight.dtype))
                if isinstance(cell, Linear) and cell.bias is not None:
                    cell.bias.set_data(weight_init_.initializer(weight_init_.Zero(),
                                                                cell.bias.shape,
                                                                cell.bias.dtype))
            elif isinstance(cell, (LayerNorm, nn.LayerNorm)):
                cell.gamma.set_data(weight_init_.initializer(weight_init_.One(),
                                                             cell.gamma.shape,
                                                             cell.gamma.dtype))
                cell.beta.set_data(weight_init_.initializer(weight_init_.Zero(),
                                                            cell.beta.shape,
                                                            cell.beta.dtype))

    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  # B L C
        x = self.avgpool(self.transpose(x, (0, 2, 1)), 2)  # B C 1
        return x

    def construct(self, x):
        x = self.forward_features(x)
        return x


class FinetuneSwin(nn.Cell):
    """finetune swim"""

    def __init__(self, **kwargs):
        super(FinetuneSwin, self).__init__()
        self.encoder = SwinTransformer(**kwargs)
        parallel_config = self.encoder.parallel_config
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.head = Linear(
            self.encoder.num_features, self.encoder.num_classes,
            weight_init=weight_init_.TruncatedNormal(sigma=2e-5),
            compute_dtype=mstype.float32).to_float(mstype.float32)
        self.head.shard(strategy_bias=((dp, mp), (mp,)), strategy_matmul=((dp, 1), (mp, 1)))

    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}

    def load_pretrained(self, params_dict):
        return load_param_into_net(self, params_dict)

    def construct(self, img):
        x = self.encoder(img)
        return self.head(x)


def swin_tiny_p4_w6(**kwargs):
    return SwinTransformer(
        image_size=192, patch_size=4, embed_dim=96, depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24], window_size=6, mlp_ratio=4, **kwargs)


def swin_tiny_p4_w7(**kwargs):
    return SwinTransformer(
        image_size=224, patch_size=4, embed_dim=96, depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24], window_size=6, mlp_ratio=4, **kwargs)


def swin_base_p4_w6(**kwargs):
    return SwinTransformer(
        patch_size=4, embed_dim=128, depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32], window_size=6, mlp_ratio=4, **kwargs)


def swin_base_p4_w7(**kwargs):
    return SwinTransformer(
        image_size=224, patch_size=4, embed_dim=128, depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32], window_size=7, mlp_ratio=4, **kwargs)


def build_swin(config):
    """build swim"""
    model = FinetuneSwin(
        parallel_config=config.parallel_config,
        moe_config=config.moe_config,
        batch_size=config.train_config.batch_size * config.device_num
        if config.parallel.parallel_mode == "semi_auto_parallel" else config.train_config.batch_size,
        image_size=config.train_config.image_size,
        patch_size=config.model.patch_size,
        in_chans=config.model.in_chans,
        num_classes=config.train_config.num_classes,
        embed_dim=config.model.embed_dim,
        depths=config.model.depth,
        num_heads=config.model.num_heads,
        window_size=config.model.window_size,
        mlp_ratio=config.model.mlp_ratio,
        qkv_bias=config.model.qkv_bias,
        qk_scale=config.model.qk_scale,
        drop_rate=config.model.drop_rate,
        drop_path_rate=config.model.drop_path_rate,
        ape=config.model.ape,
        patch_norm=config.model.patch_norm,
        patch_type=config.model.patch_type)
    return model
