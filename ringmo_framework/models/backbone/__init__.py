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
"""back bone of ringmo"""
from ringmo_framework.models.backbone.vit import Vit, FinetuneVit,\
    vit_base_p16, vit_large_p16, vit_huge_p14
from ringmo_framework.models.backbone.swin_transformer import SwinTransformer, FinetuneSwin,\
    swin_tiny_p4_w6, swin_tiny_p4_w7, swin_base_p4_w6, swin_base_p4_w7
