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
import warnings
from io import BytesIO
from PIL import Image
import numpy as np

import mindspore.dataset as de
import mindspore.common.dtype as mstype
from mindspore.dataset.vision.utils import Inter
import mindspore.dataset.vision.py_transforms as P
import mindspore.dataset.transforms.c_transforms as C2

from ringmo_framework.datasets.transforms.mixup import Mixup
from ringmo_framework.datasets.transforms.random_erasing import RandomErasing
from ringmo_framework.datasets.transforms.auto_augment import rand_augment_transform
from ringmo_framework.datasets.utils import _check_finetune_dataset_config

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_dataset(config, is_train=True):
    """build dataset"""
    data_type = config.data_type.lower()
    if is_train:
        if data_type == "mindrecord":
            train_path = os.path.join(config.train_path, config.train_ids)
            ds = de.MindDataset(train_path,
                                columns_list=config.input_columns,
                                shuffle=config.shuffle,
                                num_parallel_workers=config.num_workers,
                                num_shards=config.device_num,
                                shard_id=config.local_rank)
        elif data_type == "custom":
            ds = de.ImageFolderDataset(config.train_path,
                                       shuffle=config.shuffle,
                                       num_parallel_workers=config.num_workers,
                                       num_shards=config.device_num,
                                       shard_id=config.local_rank)
        else:
            raise NotImplementedError("Only support mindrecord or custom dataset, but got {}".format(data_type))
    else:
        batch_per_step = config.batch_size * config.device_num
        if batch_per_step < config.samples_num:
            if config.samples_num % batch_per_step == 0:
                num_padded = 0
            else:
                num_padded = batch_per_step - (config.samples_num % batch_per_step)
        else:
            num_padded = batch_per_step - config.samples_num
        print("num_padded", num_padded)
        if num_padded != 0:
            # padded_with_decode
            white_io = BytesIO()
            Image.new(
                'RGB', (config.image_size, config.image_size), (255, 255, 255)).save(white_io, 'JPEG')
            padded_sample = {
                'image': np.array(bytearray(white_io.getvalue()), dtype='uint8'),
                'label': np.array(-1, np.int32)
            }
            sample = [padded_sample for x in range(num_padded)]
            ds_pad = de.PaddedDataset(sample)
            if data_type == "mindrecord":
                eval_path = os.path.join(config.eval_path, config.eval_ids)
                ds_imagefolder = de.MindDataset(eval_path,
                                                columns_list=config.input_columns,
                                                shuffle=False,
                                                num_parallel_workers=config.num_workers)
            elif data_type == "custom":
                ds_imagefolder = de.ImageFolderDataset(config.eval_path,
                                                       num_parallel_workers=config.num_workers)
            else:
                raise NotImplementedError("Only support mindrecord or custom dataset, but got {}".format(data_type))
            ds = ds_pad + ds_imagefolder
            distribute_sampler = de.DistributedSampler(shuffle=False,
                                                       num_shards=config.device_num,
                                                       shard_id=config.local_rank)
            ds.use_sampler(distribute_sampler)
        else:
            if data_type == "mindrecord":
                eval_path = os.path.join(config.eval_path, config.eval_ids)
                ds = de.MindDataset(eval_path,
                                    columns_list=config.input_columns,
                                    shuffle=False,
                                    num_parallel_workers=config.num_workers,
                                    num_shards=config.device_num,
                                    shard_id=config.local_rank)
            elif data_type == "custom":
                ds = de.ImageFolderDataset(config.eval_path,
                                           num_parallel_workers=config.num_workers,
                                           shuffle=False,
                                           num_shards=config.device_num,
                                           shard_id=config.local_rank)
            else:
                raise NotImplementedError("Only support mindrecord or custom dataset, but got {}".format(data_type))
    return ds


def build_transforms(config, interpolation, is_train=True):
    """build transforms"""
    if is_train:
        aa_params = dict(
            translate_const=int(config.image_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in MEAN]),
        )
        assert config.auto_augment.startswith('rand')
        aa_params['interpolation'] = interpolation
        trans = [
            P.Decode(),
            P.RandomResizedCrop(config.image_size,
                                scale=(config.crop_min, 1.0),
                                ratio=(3. / 4., 4. / 3.),
                                interpolation=interpolation),
            P.RandomHorizontalFlip(prob=config.hflip),
        ]
        if config.auto_augment is None:
            trans += [P.RandomColorAdjust(brightness=config.color_jitter,
                                          contrast=config.color_jitter,
                                          saturation=config.color_jitter)]
        else:
            # trans += [ImageNetPolicyV2()]
            trans += [rand_augment_transform(config.auto_augment, aa_params)]

        trans += [
            P.ToTensor(),
            P.Normalize(mean=MEAN, std=STD),
            RandomErasing(probability=config.re_prop, mode=config.re_mode, max_count=config.re_count)
        ]

        if config.arch == "simmim":
            if not config.image_size > 32:
                trans[1] = P.RandomCrop(size=config.image_size, padding=4)

    else:
        trans = [
            P.Decode(),
            P.Resize(int(256 / 224 * config.image_size), interpolation=interpolation),
            P.CenterCrop(config.image_size),
            P.ToTensor(),
            P.Normalize(mean=MEAN, std=STD),
        ]

    return trans


def create_finetune_dataset(config, is_train=True):
    """create_dataset"""
    _check_finetune_dataset_config(config)
    dataset_config = config.finetune_dataset
    de.config.set_seed(config.seed)
    de.config.set_prefetch_size(dataset_config.prefetch_size)
    de.config.set_numa_enable(dataset_config.numa_enable)

    if hasattr(Inter, dataset_config.interpolation):
        interpolation = getattr(Inter, dataset_config.interpolation)
    else:
        interpolation = Inter.BICUBIC

    if config.auto_tune and not config.profile and is_train:
        os.makedirs(config.filepath_prefix, exist_ok=True)
        filepath_prefix = os.path.join(config.filepath_prefix, "autotune")
        de.config.set_enable_autotune(True, filepath_prefix=filepath_prefix)
        de.config.set_autotune_interval(config.autotune_per_step)

    ds = build_dataset(dataset_config, is_train)
    transforms = build_transforms(dataset_config, interpolation, is_train)
    # define map operations

    type_cast_op = C2.TypeCast(mstype.int32)
    ds = ds.map(input_columns=dataset_config.input_columns[0],
                num_parallel_workers=dataset_config.num_workers,
                operations=transforms,
                python_multiprocessing=dataset_config.python_multiprocessing)
    ds = ds.map(input_columns=dataset_config.input_columns[1],
                num_parallel_workers=dataset_config.num_workers,
                operations=type_cast_op)

    ds = ds.batch(dataset_config.batch_size, drop_remainder=True)

    if is_train and (dataset_config.mixup > 0. or dataset_config.cutmix > 0.):
        mixup_fn = Mixup(
            mixup_alpha=dataset_config.mixup, cutmix_alpha=dataset_config.cutmix,
            cutmix_minmax=None, prob=dataset_config.mixup_prob,
            switch_prob=dataset_config.switch_prob,
            label_smoothing=dataset_config.label_smoothing,
            num_classes=dataset_config.num_classes)

        ds = ds.map(operations=mixup_fn, input_columns=dataset_config.input_columns,
                    column_order=dataset_config.column_order,
                    output_columns=dataset_config.output_columns,
                    num_parallel_workers=dataset_config.num_workers)

    ds = ds.repeat(dataset_config.repeat)
    return ds
