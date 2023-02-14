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
"""create train or eval dataset."""

import os
import json

from PIL import Image

import mindspore.dataset as de
from mindspore.dataset.transforms.c_transforms import Compose
from mindspore.dataset.vision import Inter
import mindspore.dataset.vision.c_transforms as C

from ringmo_framework.datasets.utils import _check_pretrain_dataset_config
from ringmo_framework.datasets.mask.mask_policy import MaskPolicyForSim, MaskPolicyForMae, MaskPolicyForPIMask

MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]


class ImageLoader:
    """ImageLoader of datasets"""
    def __init__(self, opt_ids, data_dir=None):
        """Loading image files as a dataset generator."""
        opt_path = os.path.join(data_dir, opt_ids)
        with open(opt_path, 'r') as f_opt:
            opt_data = json.load(f_opt)
        if data_dir is not None:
            opt_data = [os.path.join(data_dir, item) for item in opt_data]

        self.opt_data = opt_data

    def __getitem__(self, index):
        out = ()
        opt_img = Image.open(self.opt_data[index]).convert("RGB")
        out = out + (opt_img,)
        return out

    def __len__(self):
        return len(self.opt_data)


def build_dataset(args):
    """build dataset"""
    if args.input_columns is None:
        args.input_columns = ["image"]

    data_type = args.data_type.lower()

    if data_type == "mindrecord":
        data_path = os.path.join(args.data_path, args.image_ids)
        dataset = de.MindDataset(data_path,
                                 columns_list=args.input_columns,
                                 num_shards=args.device_num,
                                 shard_id=args.local_rank,
                                 shuffle=args.shuffle,
                                 num_parallel_workers=args.num_workers,
                                 num_samples=args.num_samples)

    elif data_type == "custom":
        dataset = de.GeneratorDataset(source=ImageLoader(args.image_ids, data_dir=args.data_path),
                                      column_names=args.input_columns,
                                      num_shards=args.device_num,
                                      shard_id=args.local_rank,
                                      shuffle=args.shuffle,
                                      num_parallel_workers=args.num_workers,
                                      python_multiprocessing=args.python_multiprocessing)
    else:
        raise NotImplementedError("Only support mindrecord or custom dataset, but got {}".format(data_type))
    return dataset


def build_transforms(args):
    """build transforms"""
    data_type = args.data_type.lower()

    trans = [
        C.RandomResizedCrop(
            args.image_size,
            scale=(args.crop_min, 1.0),
            ratio=(3. / 4., 4. / 3.),
            interpolation=Inter.PILCUBIC),
        C.RandomHorizontalFlip(prob=args.prop),
        C.Normalize(mean=MEAN, std=STD),
        C.HWC2CHW(),
    ]

    if data_type == "mindrecord":
        trans.insert(0, C.Decode())

    trans = Compose(trans)
    return trans


def build_mask(args, ds, input_columns=None, output_columns=None, column_order=None):
    """build mask"""
    batch_size = args.batch_size
    if args.arch == 'simmim':
        if not input_columns:
            input_columns = ["image"]
        if not output_columns:
            output_columns = ["image", "mask"]
        if not column_order:
            column_order = ["image", "mask"]
        generate_mask = MaskPolicyForSim(
            input_size=args.image_size, mask_patch_size=args.mask_patch_size,
            model_patch_size=args.patch_size, mask_ratio=args.mask_ratio)
        ds = ds.map(
            operations=generate_mask, input_columns=input_columns, column_order=column_order,
            output_columns=output_columns, num_parallel_workers=args.num_workers,
            python_multiprocessing=args.python_multiprocessing)
    elif args.arch == 'mae':
        if not input_columns:
            input_columns = ["image"]
        if not output_columns:
            output_columns = ["image", "mask", "ids_restore", "unmask_index"]
        if not column_order:
            column_order = ["image", "mask", "ids_restore", "unmask_index"]
        generate_mask = MaskPolicyForMae(
            input_size=args.image_size, patch_size=args.patch_size, mask_ratio=args.mask_ratio)
        ds = ds.map(
            operations=generate_mask, input_columns=input_columns, column_order=column_order,
            output_columns=output_columns, num_parallel_workers=args.num_workers,
            python_multiprocessing=args.python_multiprocessing)
    elif args.arch == "ringmo":
        if not input_columns:
            input_columns = ["image"]
        if not output_columns:
            output_columns = ["image", "mask"]
        if not column_order:
            column_order = ["image", "mask"]
        if args.use_lbp:
            output_columns = column_order = ["image", "lbp_image", "mask"]
        generate_mask = MaskPolicyForPIMask(
            input_size=args.image_size, mask_patch_size=args.mask_patch_size,
            mask_ratio=args.mask_ratio, inside_ratio=args.inside_ratio, use_lbp=args.use_lbp)
        ds = ds.map(
            operations=generate_mask, input_columns=input_columns, column_order=column_order,
            output_columns=output_columns, num_parallel_workers=args.num_workers,
            python_multiprocessing=args.python_multiprocessing)
    else:
        raise NotImplementedError(args.arch)
    ds = ds.batch(batch_size, drop_remainder=True, num_parallel_workers=args.num_workers)
    return ds


def create_pretrain_dataset(args):
    """Create dataset for self-supervision training."""
    _check_pretrain_dataset_config(args)
    dataset_config = args.pretrain_dataset
    de.config.set_seed(args.seed)
    de.config.set_prefetch_size(dataset_config.prefetch_size)
    de.config.set_numa_enable(dataset_config.numa_enable)
    if args.auto_tune and not args.profile:
        os.makedirs(args.filepath_prefix, exist_ok=True)
        args.filepath_prefix = os.path.join(args.filepath_prefix, "autotune")
        de.config.set_enable_autotune(True, filepath_prefix=args.filepath_prefix)
        de.config.set_autotune_interval(args.autotune_per_step)

    ds = build_dataset(dataset_config)
    transforms = build_transforms(dataset_config)

    for column in dataset_config.input_columns:
        ds = ds.map(input_columns=column,
                    operations=transforms,
                    num_parallel_workers=dataset_config.num_workers,
                    python_multiprocessing=dataset_config.python_multiprocessing)

    ds = build_mask(dataset_config, ds,
                    input_columns=dataset_config.input_columns,
                    output_columns=dataset_config.output_columns,
                    column_order=dataset_config.column_order)
    ds = ds.repeat(dataset_config.repeat)
    return ds
