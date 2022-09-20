# Copyright 2021 Huawei Technologies Co., Ltd
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
"""cut image"""
import os

from PIL import Image
import numpy as np

import mindspore.dataset as de

Image.MAX_IMAGE_PIXELS = 2300000000
image_ids = 0


class CutImage:
    """cut image class"""
    def __init__(self, root_path, save_dir):
        """Loading image files as a dataset generator."""
        img_path = []
        for root, _, fs in os.walk(root_path):
            for f in fs:
                if f.endswith('.png') or f.endswith(".jpg") or f.endswith(".tif"):
                    img_path.append(os.path.join(root, f))
        self.img_path = img_path
        self.save_dir = save_dir
        self.root_path = root_path

    def __getitem__(self, index):
        img = Image.open(self.img_path[index])
        w, h = 448, 488
        global image_ids
        image_ids += int(img.size[0] / w) * int(img.size[1] / h) * 4
        cur_img_name = self.img_path[index].split('/')[-1]
        cut_save_img(img, w, h, self.save_dir, cur_img_name)
        return (np.array([0]),)

    def __len__(self):
        return len(self.img_path)


def cut_save_img(img, dw, dh, save_path, img_name):
    """cut save image"""
    w, h = img.size

    w_nums = int(w / dw) * 2
    h_nums = int(h / dh) * 2

    suf = ".png"

    imgs = (img.crop(box=(int(i * dw * 0.5), int(j * dh * 0.5), (i * 0.5 + 1) * dw, (j * 0.5 + 1) * dh))
            for i in range(w_nums - 1) for j in range(h_nums - 1))
    index_ = 0
    for img_ in imgs:
        new_img_path = os.path.join(save_path, img_name[:-4] + "_img_" + str(index_) + suf)
        if os.path.exists(new_img_path):
            continue
        img_.save(new_img_path)

        index_ += 1


def main():
    cur_path = "/mnt/aircas/pretrain/full_data/JPEGImages"  # "/mnt/aircas/pretrain/full_data/CITY-OSM" # JPEGImages deepglobe
    new_save_path = "/mnt/aircas/pretrain/new_data_448_last/split_JPEGImages"
    os.makedirs(new_save_path, exist_ok=True)
    de.config.set_multiprocessing_timeout_interval(100000)
    ds = de.GeneratorDataset(
        source=CutImage(cur_path, new_save_path),
        column_names=["image"], shuffle=False,
        num_parallel_workers=4, python_multiprocessing=False)
    ds_it = ds.create_dict_iterator()
    index_ = 0
    for d in ds_it:
        if index_ % 10 == 0:
            print(index_, d["image"].shape)
        index_ += 1
        print("total_imgs:{}".format(image_ids))

    # json.dump(image_ids, open('/mnt/aircas/pretrain/split_full_data/extra_aid/tif_cut_ids.json', 'w'))


if __name__ == "__main__":
    main()
