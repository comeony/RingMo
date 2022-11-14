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
"""prepare datasets——get image ids and output a json file"""
import argparse
import os
import json


def main(image_path, file_name):
    pretrain_ids = [os.path.join(root, f) for root, ds, fs in os.walk(image_path) for f in fs if
                    f.endswith('.png') or f.endswith(".jpg") or f.endswith(".JPEG")]
    json.dump(pretrain_ids, open(file_name, 'w'))
    print("get %d imgs." % len(pretrain_ids))
    with open("image nums", "w") as fp:
        fp.write("get %d imgs." % len(pretrain_ids))


if __name__ == "__main__":
    work_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default=None, help='image path dir')
    parser.add_argument('--file', default=None, help='json file name')

    args_ = parser.parse_args()
    if args_.image is not None and args_.file is not None:
        input_image_path = args_.image
        input_file_name = args_.file
        main(input_image_path, input_file_name)
