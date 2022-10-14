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
"""ringmo of ringmo-framework"""
from mindspore import nn
from mindspore import ops as P
from mindspore import dtype as mstype

from ringmo_framework.loss.loss import L1Loss
from ringmo_framework.models.backbone.vit import Vit
from ringmo_framework.models.backbone.swin_transformer import SwinTransformer


class SwinTransformerForRingMo(SwinTransformer):
    """swim transformer for ringmo"""
    def __init__(self, **kwargs):
        super(SwinTransformerForRingMo, self).__init__(**kwargs)
        assert self.num_classes == 0
        dp = self.parallel_config.data_parallel
        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(((dp, 1, 1),))
        self.add_pos = P.Add().shard(((dp, 1, 1), (1, 1, 1)))
        self.sub = P.Sub().shard(((), (dp, 1, 1)))
        self.multi = P.Mul().shard(((dp, 1, 1), (dp, 1, 1)))
        self.hw = int(self.final_seq ** 0.5)

    def construct(self, x, mask):
        """construct of SwinTransformerForRingMo"""
        # pylint: disable=W0221
        x = self.multi(x, self.sub(1, mask))
        x = self.patch_embed(x)

        if self.ape:
            x = self.add_pos(x, self.absolute_pos_embed)

        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.transpose(x, (0, 2, 1))
        x = self.reshape(x, (x.shape[0], x.shape[1], self.hw, self.hw))
        return x

    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}


class VisionTransformerForRingMo(Vit):
    """vision transformer for ringmo"""
    def __init__(self, **kwargs):
        super(VisionTransformerForRingMo, self).__init__(**kwargs)

        assert self.num_classes == 0
        dp = self.parallel_config.data_parallel
        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(((dp, 1, 1),))
        self.add_pos = P.Add().shard(((dp, 1, 1), (1, 1, 1)))
        self.sub = P.Sub().shard(((), (dp, 1, 1)))
        self.multi = P.Mul().shard(((dp, 1, 1), (dp, 1, 1)))
        self.hw = int(self.num_patches ** 0.5)

        self.slice = P.Slice().shard(((dp, 1, 1),))

    def construct(self, x, mask):
        """construct of VisionTransformerForRingMo"""
        # pylint: disable=W0221
        x = self.multi(x, self.sub(1, mask))
        x = self.patch_embed(x)

        batch, seq, channel = x.shape
        cls_tokens = self.tile(self.cls_tokens, (batch, 1, 1))
        x = self.cat((cls_tokens, x))
        if self.pos_embed is not None:
            x = self.add_pos(x, self.pos_embed)

        x = self.dropout(x)

        if self.rel_pos_bias:
            rel_pos_bias = self.rel_pos_bias()
            x = self.encoder(x, self.encoder_input_mask, rel_pos_bias=rel_pos_bias)
        else:
            x = self.encoder(x, self.encoder_input_mask)
        x = self.norm(x)
        x = self.slice(x, (0, 1, 0), (batch, seq, channel))  # x = x[:, 1:]
        x = self.transpose(x, (0, 2, 1))
        x = self.reshape(x, (x.shape[0], x.shape[1], self.hw, self.hw))
        return x


class RingMo(nn.Cell):
    """RingMo"""
    def __init__(self, encoder, encoder_stride, use_lbp=False, parallel_config=None):
        super(RingMo, self).__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        self.use_lbp = use_lbp
        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1

        self.decoder = nn.Conv2d(
            in_channels=self.encoder.num_features,
            out_channels=self.encoder_stride ** 2 * 3,
            kernel_size=1, has_bias=True, pad_mode='pad'
        )

        # encoder output -> [B,C,H,W]
        self.decoder.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        self.decoder.bias_add.shard(((dp, 1, 1, 1), (1,)))

        self.decoder_lbp = nn.Conv2d(
            in_channels=self.encoder.num_features,
            out_channels=self.encoder_stride ** 2 * 3,
            kernel_size=1, has_bias=True, pad_mode='pad'
        )

        # encoder output -> [B,C,H,W]
        self.decoder_lbp.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        self.decoder_lbp.bias_add.shard(((dp, 1, 1, 1), (1,)))

        self.pixelshuffle = P.DepthToSpace(self.encoder_stride).shard(((dp, 1, 1, 1),))
        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size
        self.l1_loss = L1Loss(reduction='none', parallel_config=parallel_config)

        self.expand_dim = P.ExpandDims().shard(((dp, 1, 1),))
        self.cast = P.Cast()
        self.div = P.Div().shard(((), ()))
        self.multi = P.Mul().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))

        self.sum = P.ReduceSum().shard(((dp, 1, 1, 1),))
        self.add = P.Add().shard(((), ()))

    def ringmo_loss(self, x, x_rec, lbp=None, lbp_rec=None, mask=None):
        """ringmo loss"""
        x = self.cast(x, mstype.float32)
        x_rec = self.cast(x_rec, mstype.float32)
        mask = self.cast(mask, mstype.float32)
        loss_ori_recon = self.l1_loss(x, x_rec)
        loss_ori_mask = self.mean(loss_ori_recon, mask)
        loss_lbp_mask = 0.
        if self.use_lbp:
            loss_lbp_recon = self.l1_loss(lbp, lbp_rec)
            loss_lbp_mask = self.mean(loss_lbp_recon, mask)
        loss = self.add(loss_ori_mask, loss_lbp_mask)
        return loss

    def mean(self, loss, mask):
        mul_a = self.multi(loss, mask)
        div_a = self.sum(mul_a)
        sum_b = self.sum(mask)
        div_b = self.add(sum_b, 1e-5)
        loss_mask = self.div(div_a, div_b)
        loss_mask = self.div(loss_mask, self.in_chans)
        return loss_mask

    def _check_input(self, inputs):
        if not self.use_lbp:
            return inputs[0], None, inputs[1]

        return inputs[0], inputs[1], inputs[2]

    def construct(self, *inputs):
        """construct of RingMo"""
        x_in, lbp_in, mask_in = self._check_input(inputs)

        # x -> [B,L,C]
        z = self.encoder(x_in, mask_in)
        # z -> [B,C,H,W]
        x_rec = self.decoder(z)
        # self.summary_4d("decoder_conv2d", self.decoder.weight)
        # z -> [B,C,H,W]
        x_rec = self.pixelshuffle(x_rec)

        lbp_rec = None
        if lbp_in is not None:
            lbp_rec = self.decoder_lbp(z)
            lbp_rec = self.pixelshuffle(lbp_rec)

        sim_loss = self.ringmo_loss(x_in, x_rec, lbp_in, lbp_rec, mask_in)

        return sim_loss

    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def ringmo_vit_base_p16(**kwargs):
    encoder = VisionTransformerForRingMo(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs)
    return RingMo(encoder=encoder, encoder_stride=16)


def ringmo_vit_large_p16(**kwargs):
    encoder = VisionTransformerForRingMo(patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, **kwargs)
    return RingMo(encoder=encoder, encoder_stride=16)


def ringmo_swin_tiny_p4_w6(**kwargs):
    encoder = SwinTransformerForRingMo(
        image_size=192, patch_size=4, embed_dim=96, depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24], window_size=6, mlp_ratio=4, **kwargs)
    return RingMo(encoder=encoder, encoder_stride=32)


def ringmo_swin_tiny_p4_w7(**kwargs):
    encoder = SwinTransformerForRingMo(
        image_size=224, patch_size=4, embed_dim=96, depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24], window_size=6, mlp_ratio=4, **kwargs)
    return RingMo(encoder=encoder, encoder_stride=32)


def ringmo_swin_base_p4_w6(**kwargs):
    encoder = SwinTransformerForRingMo(
        image_size=192, patch_size=4, embed_dim=128, depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32], window_size=6, mlp_ratio=4, **kwargs)
    return RingMo(encoder=encoder, encoder_stride=32)


def ringmo_swin_base_p4_w7(**kwargs):
    encoder = SwinTransformerForRingMo(
        image_size=224, patch_size=4, embed_dim=128, depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32], window_size=7, mlp_ratio=4, **kwargs)
    return RingMo(encoder=encoder, encoder_stride=32)


def build_ringmo(config):
    """build ringmo"""
    model_type = config.model.backbone
    if model_type == 'swin':
        encoder = SwinTransformerForRingMo(
            parallel_config=config.parallel_config,
            moe_config=config.moe_config,
            batch_size=config.train_config.batch_size * config.device_num
            if config.parallel.parallel_mode == "semi_auto_parallel" else config.train_config.batch_size,
            image_size=config.train_config.image_size,
            patch_size=config.model.patch_size,
            in_chans=config.model.in_chans,
            num_classes=0,
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
        encoder_stride = 32
    elif model_type == 'vit':
        encoder = VisionTransformerForRingMo(
            parallel_config=config.parallel_config,
            moe_config=config.moe_config,
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
            use_abs_pos_emb=config.model.use_abs_pos_emb,
            init_values=config.model.init_values,
            use_rel_pos_bias=config.model.use_rel_pos_bias,
            use_shared_rel_pos_bias=config.model.use_shared_rel_pos_bias,
            patch_type=config.model.patch_type)
        encoder_stride = 16
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    model = RingMo(encoder=encoder, encoder_stride=encoder_stride, parallel_config=config.parallel_config,
                   use_lbp=config.model.use_lbp)

    return model
