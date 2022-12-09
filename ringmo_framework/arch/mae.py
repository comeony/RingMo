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
"""MAE of ringmo-framework"""
from mindspore import nn
import mindspore.numpy as np
from mindspore import Tensor
from mindspore import ops as P
from mindspore import dtype as mstype
from mindspore.common.parameter import Parameter
import mindspore.common.initializer as weight_init

from ringmo_framework.loss.loss import MSELoss
from ringmo_framework.models.backbone.vit import Vit
from ringmo_framework.models.layers.patch import Patchify, UnPatchify
from ringmo_framework.models.layers.layers import LayerNorm, Linear
from ringmo_framework.models.layers.vision_transformer import VisionTransformer
from ringmo_framework.models.core.sincos_pos_embed import get_2d_sincos_pos_embed


class VisionTransformerForMae(Vit):
    """vision transformer for mae"""
    def __init__(self, mask_ratio=0.75, **kwargs):
        super().__init__(**kwargs)
        assert self.num_classes == 0

        seq_length = int((1 - mask_ratio) * self.num_patches) + 1
        self.num_masked = self.num_patches - seq_length + 1
        self.encoder_config['seq_length'] = seq_length
        self.encoder = VisionTransformer(**self.encoder_config)
        dp = self.encoder_config["parallel_config"].data_parallel
        self.encoder_input_mask = Tensor(
            np.ones((self.batch_size, seq_length, seq_length)),
            mstype.float32)
        self.seq_length = seq_length
        self.add1 = P.Add().shard(((dp, 1, 1), (1, 1, 1)))
        self.stride_slice = P.StridedSlice().shard(((1, 1, 1),))
        self.expand_dim = P.ExpandDims().shard(((dp, 1),))
        self.tile = P.Tile().shard(((dp, 1, 1),))
        self.gather = P.GatherD().shard(((dp, 1, 1), (dp, 1, 1)))
        self.reshape = P.Reshape()

        if self.pos_embed is not None:  # fix-sincos
            self.init_weights_pos()

    def init_weights_pos(self):
        """init_weights_pos of VisionTransformerForMae"""
        encoder_pos_emd = Tensor(
            get_2d_sincos_pos_embed(self.pos_embed.shape[-1],  # pylint: disable=E0203
                                    int(self.num_patches ** .5),
                                    cls_token=True),
            mstype.float32
        )
        encoder_pos_emd = P.ExpandDims()(encoder_pos_emd, 0)

        self.pos_embed = Parameter(encoder_pos_emd, name='sincos_pos_embedding', requires_grad=False)

    def construct(self, img, unmask_index):
        """construct of VisionTransformerForMae"""
        # pylint: disable=W0221
        # patch to encoder tokens and add positions
        tokens = self.patch_embed(img)

        encoder_pos_embedding = self.stride_slice(
            self.pos_embed, (0, 1, 0),
            (1, self.pos_embed.shape[1], self.pos_embed.shape[2]),
            (1, 1, 1))

        tokens = self.add1(tokens, encoder_pos_embedding)

        # get the unmasked tokens to be encoded
        unmask_index_ = self.expand_dim(unmask_index, -1)
        unmask_index = self.tile(unmask_index_, (1, 1, tokens.shape[2]))
        unmask_tokens = self.gather(tokens, 1, unmask_index)

        # cls_tokens add pos_embedding
        cls_pos_embedding = self.stride_slice(
            self.pos_embed, (0, 0, 0),
            (1, 1, self.pos_embed.shape[2]),
            (1, 1, 1))
        cls_tokens = self.tile(self.cls_tokens, (self.batch_size, 1, 1))
        cls_tokens = self.add1(cls_tokens, cls_pos_embedding)

        # concat cls_tokens
        unmask_tokens = self.cat((cls_tokens, unmask_tokens))

        # attend with vision transformer
        encoded_tokens = self.encoder(unmask_tokens, self.encoder_input_mask)

        encoded_tokens = self.norm(encoded_tokens)
        return encoded_tokens


class Mae(nn.Cell):
    """Pretrain MAE Module."""

    def __init__(self, encoder, decoder_layers=12, decoder_num_heads=16, decoder_dim=512, norm_pixel_loss=False):
        super(Mae, self).__init__()
        self.encoder = encoder
        self.norm_pixel_loss = norm_pixel_loss
        self.seq_length = encoder.seq_length
        self.use_moe = encoder.use_moe
        self.batch_size = encoder.batch_size
        self.num_masked = encoder.num_masked
        self.num_patches = encoder.num_patches

        parallel_config = encoder.encoder_config["parallel_config"]
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        tgt_seq_length = self.num_patches + 1
        patch_dim = encoder.in_chans * encoder.patch_size ** 2

        self.mask_tokens = Parameter(
            weight_init.initializer(weight_init.Normal(sigma=.02), (1, 1, decoder_dim)),
            name='mask_tokens', requires_grad=True)

        self.enc_to_dec = Linear(
            encoder.embed_dim, decoder_dim, weight_init="xavier_uniform",
            compute_dtype=mstype.float16).to_float(mstype.float16)
        self.enc_to_dec.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))

        decoder_config = encoder.encoder_config
        decoder_config['num_layers'] = decoder_layers
        decoder_config['num_heads'] = decoder_num_heads
        decoder_config['hidden_size'] = decoder_dim
        decoder_config['ffn_hidden_size'] = decoder_dim * encoder.mlp_ratio
        decoder_config['seq_length'] = tgt_seq_length
        del decoder_config['moe_config']
        self.decoder = VisionTransformer(**decoder_config)

        self.decoder_pos_embed = Parameter(
            weight_init.initializer(weight_init.TruncatedNormal(sigma=.02), (1, tgt_seq_length, decoder_dim)),
            name='decoder_pos_embedding', requires_grad=False)

        self.attention_mask = Tensor(np.ones((self.batch_size, tgt_seq_length, tgt_seq_length)), mstype.float32)

        self.to_pixels = Linear(
            decoder_dim, patch_dim, weight_init="xavier_uniform",
            compute_dtype=mstype.float16).to_float(mstype.float16)
        self.to_pixels.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))

        self.decoder_norm = LayerNorm((decoder_dim,), eps=1e-6)
        self.decoder_norm.shard(((dp, 1, 1),))

        self.patchify = Patchify(patch_size=encoder.patch_size, parallel_config=parallel_config)
        self.unpatchify = UnPatchify(
            patch_size=encoder.patch_size, seq_length=encoder.num_patches, parallel_config=parallel_config)

        self.add = P.Add().shard(((dp, 1, 1), (1, 1, 1)))
        self.cat = P.Concat(axis=1).shard(((dp, 1, 1), (dp, 1, 1)))
        self.stride_slice = P.StridedSlice().shard(((dp, 1, 1),))
        self.stride_slice4d = P.StridedSlice().shard(((1, 1, 1, 1),))
        self.gather1 = P.GatherD().shard(((dp, 1), (dp, 1)))
        self.gather2 = P.GatherD().shard(((dp, 1, 1), (dp, 1, 1)))
        self.expand_dim = P.ExpandDims().shard(((dp, 1),))
        self.tile = P.Tile().shard(((dp, 1, 1),))
        self.reshape = P.Reshape()

        self.mse_loss = MSELoss(parallel_config, norm_pixel_loss)
        self.add2 = P.Add().shard(((), ()))

        self.images_summary = P.ImageSummary().shard(((dp, 1, 1, 1),))

        self.init_weights()

    def init_weights(self):
        """init weights"""
        decoder_pos_embed = Tensor(
            get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                    int(self.num_patches ** .5),
                                    cls_token=True),
            mstype.float32
        )
        decoder_pos_embed = P.ExpandDims()(decoder_pos_embed, 0)

        self.decoder_pos_embed = Parameter(decoder_pos_embed, name='sincos_decoder_pos_embedding', requires_grad=False)
        self.init_weights_vit()

    def init_weights_vit(self):
        """ ViT weight initialization."""
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
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    def construct(self, imgs, mask, ids_restore, unmask_index):
        """construct of mae"""
        _, c, h, w = imgs.shape
        input_image = self.stride_slice4d(imgs, (0, 0, 0, 0), (1, c, h, w), (1, 1, 1, 1))
        self.images_summary("input images", input_image)
        # self.images_summary("input images", input_image.transpose(0, 2, 3, 1))

        # tokens encoder
        mask = self.gather1(mask, 1, ids_restore)

        encoder_tokens = self.encoder(imgs, unmask_index)
        patches = self.patchify(imgs)

        # project encoder to decoder dimensions,
        # if they are not equal - the paper says you can get away with a smaller dimension for decoder
        unmask_tokens = self.enc_to_dec(encoder_tokens)
        unmask_tokens = self.cast(unmask_tokens, mstype.float32)

        # mask tokens add the positions using the masked indices derived above
        mask_tokens = self.tile(self.mask_tokens, (self.batch_size, self.num_masked, 1))

        # concat the masked tokens to the decoder tokens and attend with decoder
        img_tokens = self.stride_slice(
            unmask_tokens, (0, 1, 0),
            (unmask_tokens.shape[0], unmask_tokens.shape[1], unmask_tokens.shape[2]), (1, 1, 1))
        full_tokens_ = self.cat((img_tokens, mask_tokens))
        ids_restore_copy = ids_restore
        ids_restore_ = self.expand_dim(ids_restore_copy, -1)
        ids_restore_ = self.tile(ids_restore_, (1, 1, unmask_tokens.shape[2]))
        full_tokens_ = self.gather2(full_tokens_, 1, ids_restore_)
        cls_tokens = self.stride_slice(
            unmask_tokens, (0, 0, 0),
            (unmask_tokens.shape[0], 1, unmask_tokens.shape[2]), (1, 1, 1))
        decoder_tokens = self.cat((cls_tokens, full_tokens_))

        # add position embendding for decoder tokens
        decoder_tokens = self.add(decoder_tokens, self.decoder_pos_embed)
        # decoder
        decoder_tokens = self.decoder(decoder_tokens, self.attention_mask)

        # normalize decoder tokens
        decoder_tokens = self.decoder_norm(decoder_tokens)

        # splice out the mask tokens and project to pixel values
        pred = self.to_pixels(decoder_tokens)
        pred = self.cast(pred, mstype.float32)

        pred = self.stride_slice(pred, (0, 1, 0), (pred.shape[0], pred.shape[1], pred.shape[2]), (1, 1, 1))

        reconstruct_images = self.unpatchify(pred)
        reconstruct_image = self.stride_slice4d(reconstruct_images, (0, 0, 0, 0), (1, c, h, w), (1, 1, 1, 1))
        self.images_summary("reconstruct image", reconstruct_image)

        mae_loss = self.mse_loss(pred, patches, mask)
        return mae_loss


def mae_vit_base_p16(**kwargs):
    encoder = VisionTransformerForMae(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs)
    return Mae(encoder=encoder, decoder_layers=8, decoder_num_heads=16, decoder_dim=512)


def mae_vit_large_p16(**kwargs):
    encoder = VisionTransformerForMae(patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, **kwargs)
    return Mae(encoder=encoder, decoder_layers=8, decoder_num_heads=16, decoder_dim=512)


def mae_vit_huge_p14(**kwargs):
    encoder = VisionTransformerForMae(patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, **kwargs)
    return Mae(encoder=encoder, decoder_layers=8, decoder_num_heads=16, decoder_dim=512)


def build_mae(config):
    """build mae"""
    model_type = config.model.backbone
    if model_type == 'vit':
        encoder = VisionTransformerForMae(
            parallel_config=config.parallel_config,
            moe_config=config.moe_config,
            mask_ratio=config.model.mask_ratio,
            batch_size=config.train_config.batch_size * config.device_num
            if config.parallel.parallel_mode == "semi_auto_parallel" else config.train_config.batch_size,
            image_size=config.train_config.image_size,
            patch_size=config.model.patch_size,
            in_chans=config.model.in_chans,
            num_classes=0,
            embed_dim=config.model.embed_dim,
            depth=config.model.depth,
            num_heads=config.model.num_heads,
            mlp_ratio=config.model.mlp_ratio,
            drop_rate=config.model.drop_rate,
            drop_path_rate=config.model.drop_path_rate,
            init_values=config.model.init_values,
            use_abs_pos_emb=config.model.use_abs_pos_emb,
            use_rel_pos_bias=config.model.use_rel_pos_bias,
            use_shared_rel_pos_bias=config.model.use_shared_rel_pos_bias)
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    model = Mae(
        encoder=encoder, decoder_layers=config.model.decoder_layers,
        decoder_num_heads=config.model.decoder_num_heads,
        decoder_dim=config.model.decoder_dim,
        norm_pixel_loss=config.model.norm_pixel_loss)

    return model
