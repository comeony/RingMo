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
"""setup"""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ringmo_framework",
    version="0.1",
    author="Lin-Bert",
    author_email="heqinglin4@huawei.com",
    description="ringmo-framework use to pretrain model in self-supervised for cv region.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitee.com/mindspore/ringmo-framework.git",
    packages=setuptools.find_packages(),
    requires=['numpy', 'mindspore'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux Independent",
    ],
)
