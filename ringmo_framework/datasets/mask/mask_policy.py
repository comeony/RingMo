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
"""mask policy of ringmo"""
import numpy as np
import skimage.feature as ft

from mindspore.dataset.transforms import py_transforms


class MaskPolicyForSim(py_transforms.PyTensorOperation):
    """Mask generator for simmin arch."""

    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        super(MaskPolicyForSim, self).__init__()
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self, img):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=np.int32)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        return img, mask

    def __repr__(self):
        return "Mask generator for simmin arch."


class MaskPolicyForMae(py_transforms.PyTensorOperation):
    """Mask generator for Mae arch."""

    def __init__(self, input_size=192, patch_size=4, mask_ratio=0.75):
        super(MaskPolicyForMae, self).__init__()
        assert 0 < mask_ratio < 1, \
            'masking ratio must be kept between 0 and 1'
        # seq_length
        self.num_patches = (input_size // patch_size) ** 2
        # seq masked number
        self.keep_num = int((1 - mask_ratio) * self.num_patches)

    def __call__(self, imgs):
        rand_indices = np.argsort(
            np.random.uniform(size=(self.num_patches,)), axis=0).astype(np.int32)
        ids_restore = np.argsort(rand_indices, axis=0).astype(np.int32)
        mask = np.ones((self.num_patches,)).astype(np.int32)
        mask[:self.keep_num] = 0
        unmask_index = rand_indices[:self.keep_num]
        out = (imgs, mask, ids_restore, unmask_index,)
        return out

    def __repr__(self):
        return "Mask generator for mae arch."


class MaskPolicyForPIMask(py_transforms.PyTensorOperation):
    """mask policy for PI mask"""
    def __init__(self, input_size=224, mask_patch_size=32, mask_ratio=0.6, inside_ratio=0.6, use_lbp=False):
        super(MaskPolicyForPIMask, self).__init__()
        self.use_lbp = use_lbp
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.mask_ratio = mask_ratio
        self.inside_count = mask_patch_size ** 2
        self.mask_pixel = int(np.ceil(self.inside_count * inside_ratio))

        assert self.input_size % self.mask_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        return: (N, 3, H, W)
        """
        h = w = int(x.shape[1] ** .5)
        p = self.mask_patch_size
        x = np.reshape(x, (x.shape[0], h, w, p, p, -1))
        x = np.transpose(x, (0, 5, 1, 3, 2, 4))
        x = np.reshape(x, (x.shape[0], -1, h * p, h * p))
        return x

    def __call__(self, img):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]  # mask_patch id
        mask_patch = np.zeros(self.token_count, dtype=int)
        mask_patch[mask_idx] = 1
        mask_collect = np.ones((self.token_count, self.inside_count), dtype=int)
        for value, i in zip(mask_patch, range(self.token_count)):
            mask = np.zeros(self.inside_count, dtype=int)
            if value == 1:
                inside_id = np.random.permutation(self.inside_count)[:self.mask_pixel]  # mask patch inside id
                mask[inside_id] = 1
            mask_collect[i] = mask_collect[i] * mask
        mask_collect = np.expand_dims(mask_collect, 0)
        mask = np.squeeze(self.unpatchify(mask_collect), axis=0)
        out = (img,)
        if self.use_lbp:
            lbp_img = lbp(img)
            out = out + (lbp_img,)
        out = out + (mask,)
        return out


def lbp(img):
    lbps = []
    for channel in img:
        lbp_chaneel = ft.local_binary_pattern(channel, P=8, R=2, method='uniform')
        lbp_chaneel = np.expand_dims(lbp_chaneel, 0)
        lbps.append(lbp_chaneel)
    lbp_img = np.concatenate((lbps[0], lbps[1], lbps[2]), axis=0)
    return lbp_img
