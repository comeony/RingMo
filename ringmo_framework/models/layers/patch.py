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
"""patch of ringmo"""
from mindspore import nn
from mindspore import ops as P
import mindspore.common.initializer as weight_init

from .layers import Identity, LayerNorm


class PatchEmbed(nn.Cell):
    """Construct the embeddings from patch, position embeddings."""

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_features=3,
                 out_features=768,
                 norm_layer=False,
                 patch_type="conv",
                 parallel_config=None):
        super(PatchEmbed, self).__init__()
        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.projection = build_projection(in_features, out_features, patch_size[0],
                                           parallel_config, proj_type=patch_type)
        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(((dp, 1, 1),))
        # usually not use norm
        self.norm = LayerNorm((out_features,), eps=1e-6).shard(((dp, 1, 1),)) if norm_layer else Identity()

    def construct(self, x):
        """construct"""
        # True x: bs  False x: bs * dp
        x = self.projection(x)
        b, c, h, w = x.shape
        x = self.reshape(x, (b, c, h * w))
        x = self.transpose(x, (0, 2, 1))
        x = self.norm(x)
        return x


def get_kernel_size(patch_size):
    """
        input: 2^i & i <= 5 | 14
        output: a list of kernel size
    """
    x = None
    y = None
    z = None
    ans = False
    for i in range(1, patch_size + 1):
        if patch_size % i == 0:
            x = i
            mul_y_z = patch_size // i
            for j in range(1, mul_y_z + 1):
                if mul_y_z % j == 0:
                    y = j
                    z = mul_y_z // j
                    if x >= y >= z:
                        ans = True
                        break
            if ans:
                break
    if not ans:
        raise ValueError(patch_size)
    return [x, y, z]


def build_projection(in_features, out_features, patch_size, parallel_config=None, proj_type="conv"):
    """build projection"""
    if parallel_config:
        dp = parallel_config.data_parallel
    else:
        dp = 1
    if proj_type == "conv":
        projection = nn.Conv2d(
            in_channels=in_features, out_channels=out_features,
            kernel_size=patch_size, stride=patch_size,
            weight_init=weight_init.TruncatedNormal(sigma=0.02),
            has_bias=True,
            pad_mode='pad')
        projection.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        projection.bias_add.shard(((dp, 1, 1, 1), (1,)))
    elif proj_type == "pi_conv":
        kernel_size_list = get_kernel_size(patch_size)
        stride_list = kernel_size_list
        conv1 = nn.Conv2d(in_features, out_features // 4,
                          kernel_size=kernel_size_list[0], stride=stride_list[0],
                          weight_init=weight_init.TruncatedNormal(sigma=0.02),
                          has_bias=True,
                          pad_mode='pad')
        conv1.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        conv1.bias_add.shard(((dp, 1, 1, 1), (1,)))
        bn1 = nn.BatchNorm2d(num_features=out_features // 4)
        bn1.bn_train.shard(((dp, 1, 1, 1),))
        gelu1 = nn.GELU()
        gelu1.gelu.shard(((dp, 1, 1, 1),))

        conv2 = nn.Conv2d(out_features // 4, out_features // 4,
                          kernel_size=kernel_size_list[1], stride=stride_list[1],
                          weight_init=weight_init.TruncatedNormal(sigma=0.02),
                          has_bias=True,
                          pad_mode='pad')
        conv2.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        conv2.bias_add.shard(((dp, 1, 1, 1), (1,)))
        bn2 = nn.BatchNorm2d(out_features // 4)
        bn2.bn_train.shard(((dp, 1, 1, 1),))
        gelu2 = nn.GELU()
        gelu2.gelu.shard(((dp, 1, 1, 1),))

        conv3 = nn.Conv2d(out_features // 4, out_features,
                          kernel_size=kernel_size_list[2], stride=stride_list[2],
                          weight_init=weight_init.TruncatedNormal(sigma=0.02),
                          has_bias=True,
                          pad_mode='pad')
        conv3.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        conv3.bias_add.shard(((dp, 1, 1, 1), (1,)))
        bn3 = nn.BatchNorm2d(out_features)
        bn3.bn_train.shard(((dp, 1, 1, 1),))
        projection = nn.SequentialCell(conv1, bn1, gelu1, conv2, bn2, gelu2, conv3, bn3)
    else:
        raise NotImplementedError(
            "projection: {} is not support, you can input one of [conv, pi_conv]".format(proj_type))
    return projection


class Patchify(nn.Cell):
    """Patchify"""
    def __init__(self, patch_size, parallel_config=None):
        super(Patchify, self).__init__()
        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1
        self.patch_size = patch_size
        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(((dp, 1, 1, 1, 1, 1),))

    def construct(self, img):
        p = self.patch_size
        bs, channels, h, w = img.shape
        x = self.reshape(img, (bs, channels, h // p, p, w // p, p))
        x = self.transpose(x, (0, 2, 4, 3, 5, 1))
        patches = self.reshape(x, (bs, (h // p) * (w // p), channels * p * p))
        return patches


class UnPatchify(nn.Cell):
    """UnPatchify"""
    def __init__(self, patch_size, seq_length, parallel_config=None):
        super(UnPatchify, self).__init__()
        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1
        self.p = patch_size
        self.h = self.w = int(seq_length ** .5)
        assert self.h * self.w == seq_length

        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(((dp, 1, 1, 1, 1, 1),))

    def construct(self, x):
        bs = x.shape[0]
        x = self.reshape(x, (bs, self.h, self.w, self.p, self.p, 3))
        x = self.transpose(x, (0, 5, 1, 3, 2, 4))
        images = self.reshape(x, (bs, 3, self.h * self.p, self.w * self.p))
        return images
