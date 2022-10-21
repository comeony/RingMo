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
"""mindrecord of ringmo"""
import os
import time
import json
import argparse

import numpy as np

import mindspore.mindrecord as record
import mindspore.dataset as de

big_list = []


class DataLoader:
    """data loader"""
    def __init__(self, img_ids, data_dir=None):
        """Loading image files as a dataset generator."""
        imgs_path = os.path.join(data_dir, img_ids)
        assert os.path.exists(imgs_path), "imgs_path should be real path:{}.".format(imgs_path)
        with open(imgs_path, 'r') as f:
            data = json.load(f)
        if data_dir is not None:
            data = [os.path.join(data_dir, item) for item in data]
        self.data = data

    def __getitem__(self, index):
        with open(self.data[index], 'rb') as f:
            try:
                img = f.read()
            # pylint: disable=W0703
            except Exception as e:
                print(e)

        row = {"image": img}

        try:
            writer.write_raw_data([row], parallel_writer=True)
        # pylint: disable=W0703
        except Exception as e:
            print(e)
        return (np.array([0]),)

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    work_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default=None, help='image path')
    parser.add_argument('--image_ids', default=None, help='json file for image ids')
    parser.add_argument('--save_path', default="./data/image_mindrecord", help='save path dir')
    parser.add_argument('--num_parallel_workers', default=16, help='parallel workers num')
    args = parser.parse_args()

    # 输出的MindSpore Record文件完整路径
    MINDRECORD_FILE = args.save_path
    if os.path.exists(MINDRECORD_FILE):
        os.remove(MINDRECORD_FILE)
    else:
        os.makedirs(MINDRECORD_FILE, exist_ok=True)
        print(f"makdir {MINDRECORD_FILE}")
    MINDRECORD_FILE = os.path.join(MINDRECORD_FILE, 'image.mindrecord')

    # 定义包含的字段
    cv_schema = {"image": {"type": "bytes"}}

    # 声明MindSpore Record文件格式
    writer = record.FileWriter(file_name=MINDRECORD_FILE, shard_num=20)
    writer.add_schema(cv_schema, "image")
    writer.set_page_size(1 << 26)
    ds = de.GeneratorDataset(source=DataLoader(args.image_ids, data_dir=args.image_path),
                             column_names=["image"],
                             shuffle=False, num_parallel_workers=args.num_parallel_workers,
                             python_multiprocessing=False)

    count = 0
    t0 = time.time()
    ds_it = ds.create_dict_iterator()
    for d in ds_it:
        if count % 10000 == 0:
            print(count)
        count += 1

    writer.commit()
    print(time.time() - t0)
